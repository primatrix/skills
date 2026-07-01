import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "gke_tpu.py"
SPEC = importlib.util.spec_from_file_location("gke_tpu", SCRIPT)
gke_tpu = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gke_tpu)


BASE_CONFIG = """
[gke]
project = "test-project"
cluster = "test-cluster"
zone = "us-east5-b"

[k8s]
namespace = "team-a"

[tpu]
accelerator = "tpu-v6e-slice"
topology = "4x4"
chips_per_node = 4
machine_type = "ct6e-standard-4t"
max_nodes = 4

[workload]
name = "bench"
image = "us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1"
service_account = "gcs-account"
mode = "batch"

[storage]
type = "none"

[run]
target = "script"
script = "benchmarks/foo.py"
args = ["--batch-size", "8"]
"""


class ConfigResolutionTests(unittest.TestCase):
    def test_explicit_config_path_wins_over_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            explicit = root / "custom.toml"
            explicit.write_text(BASE_CONFIG)
            (root / "gke-tpu.toml").write_text(BASE_CONFIG.replace("bench", "default"))

            resolved = gke_tpu.resolve_config_path(root, config_path=str(explicit), profile=None)

            self.assertEqual(resolved, explicit)

    def test_profile_uses_configs_gke_tpu_profile_toml(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            profile_path = root / "configs" / "gke-tpu" / "v6e-dev.toml"
            profile_path.parent.mkdir(parents=True)
            profile_path.write_text(BASE_CONFIG)

            resolved = gke_tpu.resolve_config_path(root, config_path=None, profile="v6e-dev")

            self.assertEqual(resolved, profile_path)


class PlanTests(unittest.TestCase):
    def test_nodepool_plan_derives_context_nodepool_and_confirmation(self):
        config = gke_tpu.load_config_text(BASE_CONFIG)

        plan = gke_tpu.plan_nodepool(config)

        self.assertTrue(plan["ok"])
        data = plan["data"]
        self.assertEqual(data["context"], "gke_test-project_us-east5-b_test-cluster")
        self.assertEqual(data["namespace"], "team-a")
        self.assertEqual(data["hosts"], 4)
        self.assertEqual(data["nodepool"], "bench-4x4-np")
        self.assertEqual(data["nodepool_source"], "derived")
        self.assertIn("nodepool was derived", data["warnings"][0])
        command = next(cmd for cmd in data["commands"] if cmd["id"] == "create_nodepool")
        self.assertEqual(command["id"], "create_nodepool")
        self.assertIn("gcloud", command["argv"][0])
        self.assertIn("--placement-policy=bench-4x4-np-policy", command["argv"])
        self.assertEqual(
            command["requires_confirmation"],
            "CREATE nodepool bench-4x4-np in cluster test-cluster",
        )

    def test_delete_workload_plan_uses_names_not_tmp_manifest(self):
        config = gke_tpu.load_config_text(BASE_CONFIG)

        plan = gke_tpu.plan_delete_workload(config)

        self.assertTrue(plan["ok"])
        data = plan["data"]
        self.assertEqual(
            data["resources"],
            ["job/bench", "svc/bench-headless-svc", "configmap/bench-launcher"],
        )
        for command in data["commands"]:
            self.assertIn("--context", command["argv"])
            self.assertIn("-n", command["argv"])
            self.assertIn("--ignore-not-found", command["argv"])
        self.assertEqual(
            data["requires_confirmation"],
            "DELETE workload bench in namespace team-a",
        )


class RenderTests(unittest.TestCase):
    def test_render_multi_host_batch_script_writes_multidoc_job_manifest(self):
        config = gke_tpu.load_config_text(BASE_CONFIG)
        with tempfile.TemporaryDirectory() as tmp:
            result = gke_tpu.render_workload(config, out_dir=Path(tmp))

            self.assertTrue(result["ok"])
            manifest_path = Path(result["data"]["writes"][0]["path"])
            manifest = manifest_path.read_text()

        self.assertIn("kind: ConfigMap", manifest)
        self.assertIn("apiVersion: v1\nkind: ConfigMap", manifest)
        self.assertIn("name: bench-launcher", manifest)
        self.assertIn("kind: Service", manifest)
        self.assertIn("name: bench-headless-svc", manifest)
        self.assertIn("apiVersion: batch/v1\nkind: Job", manifest)
        self.assertIn("kind: Job", manifest)
        self.assertNotIn("kind: Pod", manifest)
        self.assertIn("parallelism: 4", manifest)
        self.assertIn("completions: 4", manifest)
        self.assertIn("      subdomain: bench-headless-svc", manifest)
        self.assertIn('command: ["python3", "-u", "/opt/gke-tpu/launcher.py"]', manifest)
        self.assertIn("jax.distributed.initialize()", manifest)
        self.assertIn("benchmarks/foo.py", manifest)

    def test_render_single_host_command_has_no_service_or_launcher(self):
        single_host = BASE_CONFIG.replace('topology = "4x4"', 'topology = "2x2"')
        single_host = single_host.replace('max_nodes = 4', 'max_nodes = 1')
        single_host = single_host.replace(
            '[run]\ntarget = "script"\nscript = "benchmarks/foo.py"\nargs = ["--batch-size", "8"]',
            '[run]\ntarget = "command"\ncommand = ["bash", "-lc", "python benchmarks/foo.py"]',
        )
        config = gke_tpu.load_config_text(single_host)

        with tempfile.TemporaryDirectory() as tmp:
            result = gke_tpu.render_workload(config, out_dir=Path(tmp))
            manifest = Path(result["data"]["writes"][0]["path"]).read_text()

        self.assertNotIn("kind: Service", manifest)
        self.assertNotIn("kind: ConfigMap", manifest)
        self.assertIn("parallelism: 1", manifest)
        self.assertIn('command: ["bash", "-lc", "python benchmarks/foo.py"]', manifest)

    def test_render_storage_modes(self):
        gcsfuse = BASE_CONFIG.replace(
            '[storage]\ntype = "none"',
            '[storage]\ntype = "gcsfuse"\nmount_path = "/models"\n\n[storage.gcsfuse]\nbucket = "model-bucket"',
        )
        pvc = BASE_CONFIG.replace(
            '[storage]\ntype = "none"',
            '[storage]\ntype = "pvc"\nmount_path = "/models"\n\n[storage.pvc]\nname = "models-pvc"\nread_only = true\ngcsfuse_backed = true',
        )

        with tempfile.TemporaryDirectory() as tmp:
            gcs_manifest = Path(
                gke_tpu.render_workload(gke_tpu.load_config_text(gcsfuse), Path(tmp))["data"]["writes"][0]["path"]
            ).read_text()
            pvc_manifest = Path(
                gke_tpu.render_workload(gke_tpu.load_config_text(pvc), Path(tmp))["data"]["writes"][0]["path"]
            ).read_text()

        self.assertIn('gke-gcsfuse/volumes: "true"', gcs_manifest)
        self.assertIn("bucketName: model-bucket", gcs_manifest)
        self.assertIn("claimName: models-pvc", pvc_manifest)
        self.assertIn("readOnly: true", pvc_manifest)
        self.assertIn('gke-gcsfuse/volumes: "true"', pvc_manifest)


class ValidationTests(unittest.TestCase):
    def test_old_repo_config_is_rejected_as_breaking_change(self):
        config = gke_tpu.load_config_text(BASE_CONFIG + "\n[repo]\ngit_url = \"https://example.com/repo.git\"\n")

        result = gke_tpu.validate_config(config)

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["code"], "unsupported_repo_config")


if __name__ == "__main__":
    unittest.main()
