import Cocoa
import ApplicationServices
let pid: pid_t = NSRunningApplication.runningApplications(withBundleIdentifier: "com.openai.codex").first?.processIdentifier
  ?? NSWorkspace.shared.runningApplications.first(where: { $0.localizedName == "ChatGPT" && $0.activationPolicy == .regular })?.processIdentifier ?? -1
if pid < 0 { print("ChatGPT app not running"); exit(1) }
let app = AXUIElementCreateApplication(pid)
AXUIElementSetAttributeValue(app, "AXManualAccessibility" as CFString, kCFBooleanTrue)
func attr(_ e: AXUIElement, _ a: String) -> AnyObject? { var v: AnyObject?; AXUIElementCopyAttributeValue(e, a as CFString, &v); return v }
func label(_ e: AXUIElement) -> String { (attr(e, kAXTitleAttribute) as? String ?? "") + "|" + (attr(e, kAXDescriptionAttribute) as? String ?? "") + "|" + String((attr(e, kAXValueAttribute) as? String ?? "").prefix(70)) }
var hits: [(AXUIElement, String)] = []; var n = 0
func walk(_ e: AXUIElement, _ d: Int) {
  if d > 90 { return }; n += 1
  let role = attr(e, kAXRoleAttribute) as? String ?? ""
  let t = label(e)
  if t.contains("引导") || t.contains("Claude → Codex") || t.contains("与 Codex 协作") { hits.append((e, role + " " + t)) }
  for key in [kAXChildrenAttribute, "AXChildrenInNavigationOrder"] {
    if let kids = attr(e, key) as? [AXUIElement], kids.count > 0 { for k in kids { walk(k, d+1) }; break }
  }
}
// Chromium only exposes web content to AX while the window is rendered: activate the app if needed,
// remember the previous frontmost app, and restore it afterwards.
let previous = NSWorkspace.shared.frontmostApplication
let target = NSRunningApplication(processIdentifier: pid)
var activated = false
func countNodes(_ e: AXUIElement, _ d: Int) -> Int {
  if d > 30 { return 1 }
  let kids = (attr(e, kAXChildrenAttribute) as? [AXUIElement]) ?? []
  return 1 + kids.reduce(0) { $0 + countNodes($1, d+1) }
}
let initialWindows = (attr(app, kAXWindowsAttribute) as? [AXUIElement]) ?? []
var total = initialWindows.reduce(0, { $0 + countNodes($1, 0) })
var tries = 0
while total < 200 && tries < 6 {
  // re-assert accessibility and bring the app forward until Chromium publishes its web AX tree
  AXUIElementSetAttributeValue(app, "AXManualAccessibility" as CFString, kCFBooleanTrue)
  AXUIElementSetAttributeValue(app, "AXEnhancedUserInterface" as CFString, kCFBooleanTrue)
  target?.activate(options: [.activateAllWindows])
  activated = true
  Thread.sleep(forTimeInterval: 1.0)
  let ws = (attr(app, kAXWindowsAttribute) as? [AXUIElement]) ?? []
  total = ws.reduce(0, { $0 + countNodes($1, 0) })
  tries += 1
}
defer { if activated, let p = previous, p.processIdentifier != pid { p.activate(options: []) } }
var windows: [AXUIElement] = []
for _ in 0..<10 {
  windows = (attr(app, kAXWindowsAttribute) as? [AXUIElement]) ?? []
  if let f = attr(app, kAXFocusedWindowAttribute) { windows.append(f as! AXUIElement) }
  if !windows.isEmpty { break }; Thread.sleep(forTimeInterval: 0.5)
}
print("windows", windows.count)
for w in windows { walk(w, 0); if !hits.isEmpty { break } }
print("nodes", n)
for (i, h) in hits.enumerated() { print(i, h.1) }
if CommandLine.arguments.contains("press") {
  for (e, desc) in hits where desc.hasPrefix("AXButton") && desc.contains("引导") {
    let r = AXUIElementPerformAction(e, kAXPressAction as CFString); print("press", r.rawValue, desc); Thread.sleep(forTimeInterval: 1.2)
  }
}
