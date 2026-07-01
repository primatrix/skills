#!/usr/bin/env python3
"""Pure planning/rendering helpers for the gke-tpu skill."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import textwrap
import tomllib
from pathlib import Path
from typing import Any


class GkeTpuError(Exception):
    def __init__(self, code: str, message: str, suggested_patch: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.suggested_patch = suggested_patch

    def to_json(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error": {
                "code": self.code,
                "message": self.message,
                "suggested_patch": self.suggested_patch,
            },
        }


def ok(action: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "action": action, "data": data}


def resolve_config_path(cwd: Path, config_path: str | None, profile: str | None) -> Path:
    if config_path:
        path = Path(config_path)
        if not path.is_absolute():
            path = cwd / path
        if not path.exists():
            raise GkeTpuError("missing_config", f"Config file does not exist: {path}")
        return path

    if profile:
        path = cwd / "configs" / "gke-tpu" / f"{profile}.toml"
        if not path.exists():
            raise GkeTpuError("missing_config", f"Profile config does not exist: {path}")
        return path

    for candidate in (
        cwd / "gke-tpu.toml",
        cwd / "configs" / "gke-tpu" / "default.toml",
    ):
        if candidate.exists():
            return candidate

    raise GkeTpuError(
        "missing_config",
        "No config found. Pass --config, --profile, or create gke-tpu.toml.",
    )


def load_config_text(text: str) -> dict[str, Any]:
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise GkeTpuError("invalid_toml", str(exc)) from exc


def load_config(path: Path) -> dict[str, Any]:
    return load_config_text(path.read_text())


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if "repo" in config:
        return GkeTpuError(
            "unsupported_repo_config",
            "[repo] config is no longer supported. Put code in the image, mounted storage, or command-time git clone.",
        ).to_json()

    try:
        _model(config)
    except GkeTpuError as exc:
        return exc.to_json()
    return ok("validate", {"warnings": _warnings(config)})


def _required(config: dict[str, Any], table: str, key: str) -> Any:
    value = config.get(table, {}).get(key)
    if value in (None, ""):
        raise GkeTpuError("missing_field", f"Missing required field [{table}].{key}")
    return value


def _topology_chips(topology: str) -> int:
    try:
        dims = [int(part) for part in topology.split("x")]
    except ValueError as exc:
        raise GkeTpuError("invalid_topology", f"Invalid topology: {topology}") from exc
    if not dims or any(dim <= 0 for dim in dims):
        raise GkeTpuError("invalid_topology", f"Invalid topology: {topology}")
    return math.prod(dims)


def _context(project: str, zone: str, cluster: str, k8s: dict[str, Any]) -> str:
    explicit = k8s.get("context", "")
    return explicit or f"gke_{project}_{zone}_{cluster}"


def _workload_command(config: dict[str, Any], hosts: int) -> tuple[list[str], bool]:
    workload = config.get("workload", {})
    mode = workload.get("mode", "batch")
    run = config.get("run", {})

    if mode not in ("batch", "interactive"):
        raise GkeTpuError("invalid_workload_mode", "workload.mode must be batch or interactive")

    if mode == "interactive":
        return workload.get("command", ["sleep", "infinity"]), False

    target = run.get("target", "command")
    if target in ("script", "module"):
        return ["python3", "-u", "/opt/gke-tpu/launcher.py"], True
    if target == "command":
        command = run.get("command")
        if not isinstance(command, list) or not command:
            raise GkeTpuError("missing_run_command", "[run].command is required when target = command")
        return command, False
    raise GkeTpuError("invalid_run_target", "run.target must be script, module, or command")


def _distributed_init(config: dict[str, Any], hosts: int) -> bool:
    run = config.get("run", {})
    if "distributed_init" in run:
        return bool(run["distributed_init"])
    return hosts > 1 and run.get("target", "command") in ("script", "module")


def _warnings(config: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    tpu = config.get("tpu", {})
    if not tpu.get("nodepool"):
        workload = config.get("workload", {}).get("name", "workload")
        topology = tpu.get("topology", "topology")
        warnings.append(f"nodepool was derived as {workload}-{topology}-np; set [tpu].nodepool for shared clusters")
    return warnings


def _model(config: dict[str, Any]) -> dict[str, Any]:
    if "repo" in config:
        raise GkeTpuError(
            "unsupported_repo_config",
            "[repo] config is no longer supported. Put code in the image, mounted storage, or command-time git clone.",
        )

    project = _required(config, "gke", "project")
    cluster = _required(config, "gke", "cluster")
    zone = _required(config, "gke", "zone")
    namespace = config.get("k8s", {}).get("namespace", "default")
    context = _context(project, zone, cluster, config.get("k8s", {}))

    topology = _required(config, "tpu", "topology")
    chips_per_node = int(_required(config, "tpu", "chips_per_node"))
    chips = _topology_chips(topology)
    if chips % chips_per_node != 0:
        raise GkeTpuError("invalid_hosts", "topology chips must be divisible by tpu.chips_per_node")
    hosts = chips // chips_per_node

    workload_name = _required(config, "workload", "name")
    nodepool = config.get("tpu", {}).get("nodepool") or f"{workload_name}-{topology}-np"
    nodepool_source = "explicit" if config.get("tpu", {}).get("nodepool") else "derived"
    command, needs_launcher = _workload_command(config, hosts)

    return {
        "project": project,
        "cluster": cluster,
        "zone": zone,
        "region": zone.rsplit("-", 1)[0],
        "context": context,
        "namespace": namespace,
        "accelerator": _required(config, "tpu", "accelerator"),
        "topology": topology,
        "chips": chips,
        "chips_per_node": chips_per_node,
        "hosts": hosts,
        "machine_type": _required(config, "tpu", "machine_type"),
        "max_nodes": int(config.get("tpu", {}).get("max_nodes", hosts)),
        "reservation": config.get("tpu", {}).get("reservation", ""),
        "nodepool": nodepool,
        "nodepool_source": nodepool_source,
        "workload": workload_name,
        "image": _required(config, "workload", "image"),
        "service_account": config.get("workload", {}).get("service_account", ""),
        "mode": config.get("workload", {}).get("mode", "batch"),
        "command": command,
        "needs_launcher": needs_launcher,
        "distributed_init": _distributed_init(config, hosts),
        "storage": config.get("storage", {"type": "none"}),
        "run": config.get("run", {}),
        "warnings": _warnings(config),
    }


def _gcloud_base(model: dict[str, Any]) -> list[str]:
    return [
        "--cluster",
        model["cluster"],
        "--location",
        model["zone"],
        "--project",
        model["project"],
    ]


def plan_nodepool(config: dict[str, Any]) -> dict[str, Any]:
    try:
        model = _model(config)
    except GkeTpuError as exc:
        return exc.to_json()

    commands: list[dict[str, Any]] = []
    if model["hosts"] > 1:
        commands.append(
            {
                "id": "create_workload_policy",
                "argv": [
                    "gcloud",
                    "compute",
                    "resource-policies",
                    "create",
                    "workload-policy",
                    f"{model['nodepool']}-policy",
                    "--type",
                    "HIGH_THROUGHPUT",
                    "--accelerator-topology",
                    model["topology"],
                    "--project",
                    model["project"],
                    "--region",
                    model["region"],
                ],
                "requires_confirmation": f"CREATE nodepool {model['nodepool']} in cluster {model['cluster']}",
            }
        )

    nodepool_argv = [
        "gcloud",
        "container",
        "node-pools",
        "create",
        model["nodepool"],
        *_gcloud_base(model),
        f"--machine-type={model['machine_type']}",
    ]
    if model["reservation"]:
        nodepool_argv += [
            "--reservation-affinity=specific",
            f"--reservation={model['reservation']}",
            f"--num-nodes={model['hosts']}",
        ]
    else:
        nodepool_argv += [
            "--num-nodes=0",
            "--enable-autoscaling",
            "--total-min-nodes=0",
            f"--total-max-nodes={model['max_nodes']}",
        ]
    if model["hosts"] > 1:
        nodepool_argv.append(f"--placement-policy={model['nodepool']}-policy")

    commands.append(
        {
            "id": "create_nodepool",
            "argv": nodepool_argv,
            "requires_confirmation": f"CREATE nodepool {model['nodepool']} in cluster {model['cluster']}",
        }
    )

    return ok(
        "plan-nodepool",
        {
            **_public_model(model),
            "commands": commands,
            "requires_confirmation": f"CREATE nodepool {model['nodepool']} in cluster {model['cluster']}",
        },
    )


def plan_delete_nodepool(config: dict[str, Any]) -> dict[str, Any]:
    try:
        model = _model(config)
    except GkeTpuError as exc:
        return exc.to_json()

    commands = [
        {
            "id": "delete_nodepool",
            "argv": [
                "gcloud",
                "container",
                "node-pools",
                "delete",
                model["nodepool"],
                *_gcloud_base(model),
                "--quiet",
            ],
            "requires_confirmation": f"DELETE nodepool {model['nodepool']} from cluster {model['cluster']}",
        }
    ]
    if model["hosts"] > 1:
        commands.append(
            {
                "id": "delete_workload_policy",
                "argv": [
                    "gcloud",
                    "compute",
                    "resource-policies",
                    "delete",
                    f"{model['nodepool']}-policy",
                    "--project",
                    model["project"],
                    "--region",
                    model["region"],
                    "--quiet",
                ],
                "requires_confirmation": f"DELETE nodepool {model['nodepool']} from cluster {model['cluster']}",
            }
        )
    return ok(
        "delete-nodepool-plan",
        {
            **_public_model(model),
            "commands": commands,
            "requires_confirmation": f"DELETE nodepool {model['nodepool']} from cluster {model['cluster']}",
        },
    )


def plan_delete_workload(config: dict[str, Any]) -> dict[str, Any]:
    try:
        model = _model(config)
    except GkeTpuError as exc:
        return exc.to_json()

    resources = [f"job/{model['workload']}"]
    if model["hosts"] > 1:
        resources.append(f"svc/{model['workload']}-headless-svc")
    if model["needs_launcher"]:
        resources.append(f"configmap/{model['workload']}-launcher")

    commands = [
        {
            "id": f"delete_{resource.split('/')[0]}",
            "argv": _kubectl(model, ["delete", resource, "--ignore-not-found"]),
            "requires_confirmation": f"DELETE workload {model['workload']} in namespace {model['namespace']}",
        }
        for resource in resources
    ]
    return ok(
        "delete-workload-plan",
        {
            **_public_model(model),
            "resources": resources,
            "commands": commands,
            "requires_confirmation": f"DELETE workload {model['workload']} in namespace {model['namespace']}",
        },
    )


def render_workload(config: dict[str, Any], out_dir: Path | None = None) -> dict[str, Any]:
    try:
        model = _model(config)
        manifest = _render_manifest(model)
    except GkeTpuError as exc:
        return exc.to_json()

    base = out_dir or Path(tempfile.gettempdir()) / "gke-tpu"
    workload_dir = base / model["workload"]
    workload_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = workload_dir / "workload.yaml"
    manifest_path.write_text(manifest)

    return ok(
        "render-workload",
        {
            **_public_model(model),
            "writes": [{"kind": "manifest", "path": str(manifest_path)}],
            "commands": [
                {
                    "id": "apply_workload",
                    "argv": _kubectl(model, ["apply", "-f", str(manifest_path)]),
                    "requires_confirmation": f"APPLY workload {model['workload']} in namespace {model['namespace']}",
                }
            ],
            "requires_confirmation": f"APPLY workload {model['workload']} in namespace {model['namespace']}",
        },
    )


def init_template() -> dict[str, Any]:
    template = textwrap.dedent(
        """
        [gke]
        project = "your-gcp-project"
        cluster = "your-cluster"
        zone = "us-east5-b"

        [k8s]
        namespace = "default"
        context = "" # optional; defaults to gke_<project>_<zone>_<cluster>

        [tpu]
        accelerator = "tpu-v6e-slice"
        topology = "4x4"
        chips_per_node = 4
        machine_type = "ct6e-standard-4t"
        max_nodes = 4
        nodepool = "" # optional; derived from workload name and topology when empty
        reservation = ""

        [workload]
        name = "my-workload"
        image = "us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1"
        service_account = "gcs-account"
        mode = "batch" # batch | interactive

        [storage]
        type = "none" # none | gcsfuse | pvc

        [run]
        target = "command" # command | script | module
        command = ["bash", "-lc", "python train.py"]
        """
    ).strip() + "\n"
    return ok("init", {"template": template})


def _public_model(model: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "project",
        "cluster",
        "zone",
        "context",
        "namespace",
        "accelerator",
        "topology",
        "chips",
        "chips_per_node",
        "hosts",
        "machine_type",
        "max_nodes",
        "reservation",
        "nodepool",
        "nodepool_source",
        "workload",
        "mode",
        "warnings",
    )
    return {key: model[key] for key in keys}


def _kubectl(model: dict[str, Any], rest: list[str]) -> list[str]:
    return ["kubectl", "--context", model["context"], "-n", model["namespace"], *rest]


def _render_manifest(model: dict[str, Any]) -> str:
    docs: list[str] = []
    if model["needs_launcher"]:
        docs.append(_render_launcher_configmap(model))
    if model["hosts"] > 1:
        docs.append(_render_service(model))
    docs.append(_render_job(model))
    return "\n---\n".join(docs).rstrip() + "\n"


def _render_launcher_configmap(model: dict[str, Any]) -> str:
    launcher = _launcher_source(model)
    lines = [
        "apiVersion: v1",
        "kind: ConfigMap",
        "metadata:",
        f"  name: {model['workload']}-launcher",
        "data:",
        "  launcher.py: |",
    ]
    lines.extend(f"    {line}" if line else "" for line in launcher.splitlines())
    return "\n".join(lines)


def _launcher_source(model: dict[str, Any]) -> str:
    run = model["run"]
    args = run.get("args", [])
    target = run.get("target", "command")
    init_line = "jax.distributed.initialize()" if model["distributed_init"] else "# distributed init disabled"
    if target == "script":
        body = f'runpy.run_path({run["script"]!r}, run_name="__main__")'
        argv0 = run["script"]
    elif target == "module":
        body = f'runpy.run_module({run["module"]!r}, run_name="__main__", alter_sys=True)'
        argv0 = run["module"]
    else:
        raise GkeTpuError("invalid_launcher_target", "launcher supports script or module targets")

    return textwrap.dedent(
        f"""
        #!/usr/bin/env python3
        import runpy
        import sys

        import jax

        {init_line}
        print(f"[Process {{jax.process_index()}}] ready: {{jax.device_count()}} devices", flush=True)
        sys.argv = {[argv0, *args]!r}
        {body}
        """
    ).strip()


def _render_service(model: dict[str, Any]) -> str:
    return textwrap.dedent(
        f"""
        apiVersion: v1
        kind: Service
        metadata:
          name: {model['workload']}-headless-svc
        spec:
          clusterIP: None
          selector:
            job-name: {model['workload']}
        """
    ).strip()


def _render_job(model: dict[str, Any]) -> str:
    annotations = _pod_annotations(model)
    lines = [
        "apiVersion: batch/v1",
        "kind: Job",
        "metadata:",
        f"  name: {model['workload']}",
        "spec:",
        "  completionMode: Indexed",
        f"  parallelism: {model['hosts']}",
        f"  completions: {model['hosts']}",
        "  backoffLimit: 0",
        "  template:",
    ]
    if annotations:
        lines += ["    metadata:", "      annotations:"]
        lines.extend(f'        {key}: "{value}"' for key, value in annotations.items())
    else:
        lines.append("    metadata: {}")
    lines.append("    spec:")
    if model["hosts"] > 1:
        lines.append(f"      subdomain: {model['workload']}-headless-svc")
    lines.append("      restartPolicy: Never")
    if model["service_account"]:
        lines.append(f"      serviceAccountName: {model['service_account']}")
    lines += [
        "      nodeSelector:",
        f"        cloud.google.com/gke-tpu-accelerator: {model['accelerator']}",
        f"        cloud.google.com/gke-tpu-topology: {model['topology']}",
        f"        cloud.google.com/gke-nodepool: {model['nodepool']}",
        "      containers:",
        f"      - name: {model['workload']}",
        f"        image: {model['image']}",
        f"        command: {json.dumps(model['command'])}",
        "        resources:",
        "          requests:",
        f"            google.com/tpu: {model['chips_per_node']}",
        "          limits:",
        f"            google.com/tpu: {model['chips_per_node']}",
    ]
    volume_mounts = _volume_mount_lines(model)
    if volume_mounts:
        lines.append("        volumeMounts:")
        lines.extend(_indent_lines(volume_mounts, 10))
    volumes = _volume_lines(model)
    if volumes:
        lines.append("      volumes:")
        lines.extend(_indent_lines(volumes, 8))
    return "\n".join(lines)


def _pod_annotations(model: dict[str, Any]) -> dict[str, str]:
    storage = model["storage"]
    storage_type = storage.get("type", "none")
    if storage_type == "gcsfuse":
        return {"gke-gcsfuse/volumes": "true"}
    if storage_type == "pvc" and storage.get("pvc", {}).get("gcsfuse_backed", False):
        return {"gke-gcsfuse/volumes": "true"}
    return {}


def _volume_mount_lines(model: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if model["needs_launcher"]:
        lines += [
            "- name: launcher",
            "  mountPath: /opt/gke-tpu",
            "  readOnly: true",
        ]

    storage = model["storage"]
    storage_type = storage.get("type", "none")
    if storage_type in ("gcsfuse", "pvc"):
        _require_storage_mount(storage)
        lines += [
            "- name: model-storage",
            f"  mountPath: {storage['mount_path']}",
            f"  readOnly: {str(_storage_read_only(storage)).lower()}",
        ]

    return lines


def _volume_lines(model: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if model["needs_launcher"]:
        lines += [
            "- name: launcher",
            "  configMap:",
            f"    name: {model['workload']}-launcher",
        ]

    storage = model["storage"]
    storage_type = storage.get("type", "none")
    if storage_type == "gcsfuse":
        gcsfuse = storage.get("gcsfuse", {})
        bucket = gcsfuse.get("bucket")
        if not bucket:
            raise GkeTpuError("missing_storage_bucket", "[storage.gcsfuse].bucket is required")
        mount_options = gcsfuse.get("mount_options", "")
        option_line = f'\n          mountOptions: "{mount_options}"' if mount_options else ""
        lines += [
            "- name: gke-gcsfuse-cache",
            "  emptyDir:",
            "    medium: Memory",
            "- name: model-storage",
            "  csi:",
            "    driver: gcsfuse.csi.storage.gke.io",
            "    readOnly: false",
            "    volumeAttributes:",
            '      skipCSIBucketAccessCheck: "true"',
            '      gcsfuseMetadataPrefetchOnMount: "true"',
            f"      bucketName: {bucket}",
        ]
        if option_line:
            lines.append(f'      mountOptions: "{mount_options}"')
    elif storage_type == "pvc":
        pvc = storage.get("pvc", {})
        name = pvc.get("name")
        if not name:
            raise GkeTpuError("missing_storage_pvc", "[storage.pvc].name is required")
        if pvc.get("gcsfuse_backed", False):
            lines += [
                "- name: gke-gcsfuse-cache",
                "  emptyDir:",
                "    medium: Memory",
            ]
        lines += [
            "- name: model-storage",
            "  persistentVolumeClaim:",
            f"    claimName: {name}",
            f"    readOnly: {str(_storage_read_only(storage)).lower()}",
        ]
    elif storage_type != "none":
        raise GkeTpuError("invalid_storage_type", "storage.type must be none, gcsfuse, or pvc")

    return lines


def _storage_read_only(storage: dict[str, Any]) -> bool:
    if storage.get("type") == "pvc":
        return bool(storage.get("pvc", {}).get("read_only", False))
    return False


def _require_storage_mount(storage: dict[str, Any]) -> None:
    if not storage.get("mount_path"):
        raise GkeTpuError("missing_storage_mount_path", "[storage].mount_path is required")


def _indent_lines(lines: list[str], spaces: int) -> list[str]:
    prefix = " " * spaces
    return [prefix + line if line else line for line in lines]


def _load_from_args(args: argparse.Namespace) -> dict[str, Any]:
    path = resolve_config_path(Path.cwd(), args.config, args.profile)
    return load_config(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan and render GKE TPU workloads as JSON.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_config_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config")
        p.add_argument("--profile")

    sub.add_parser("init")
    for name in ("validate", "plan-nodepool", "delete-workload-plan", "delete-nodepool-plan"):
        add_config_flags(sub.add_parser(name))
    render = sub.add_parser("render-workload")
    add_config_flags(render)
    render.add_argument("--out-dir")

    args = parser.parse_args(argv)

    try:
        if args.command == "init":
            result = init_template()
        elif args.command == "validate":
            result = validate_config(_load_from_args(args))
        elif args.command == "plan-nodepool":
            result = plan_nodepool(_load_from_args(args))
        elif args.command == "delete-workload-plan":
            result = plan_delete_workload(_load_from_args(args))
        elif args.command == "delete-nodepool-plan":
            result = plan_delete_nodepool(_load_from_args(args))
        elif args.command == "render-workload":
            out_dir = Path(args.out_dir) if args.out_dir else None
            result = render_workload(_load_from_args(args), out_dir=out_dir)
        else:
            raise GkeTpuError("unknown_command", args.command)
    except GkeTpuError as exc:
        result = exc.to_json()

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
