import Cocoa
import ApplicationServices
let pid = NSRunningApplication.runningApplications(withBundleIdentifier: "com.openai.codex").first!.processIdentifier
let app = AXUIElementCreateApplication(pid)
func attr(_ e: AXUIElement, _ a: String) -> AnyObject? { var v: AnyObject?; AXUIElementCopyAttributeValue(e, a as CFString, &v); return v }
let r1 = AXUIElementSetAttributeValue(app, "AXManualAccessibility" as CFString, kCFBooleanTrue)
let r2 = AXUIElementSetAttributeValue(app, "AXEnhancedUserInterface" as CFString, kCFBooleanTrue)
print("set", r1.rawValue, r2.rawValue)
func count(_ e: AXUIElement, _ d: Int) -> Int { if d > 40 {return 1}; return 1 + ((attr(e, kAXChildrenAttribute) as? [AXUIElement]) ?? []).reduce(0){$0+count($1,d+1)} }
for i in 0..<10 {
  _ = attr(app, kAXFocusedUIElementAttribute)
  let ws = (attr(app, kAXWindowsAttribute) as? [AXUIElement]) ?? []
  let n = ws.reduce(0){$0+count($1,0)}
  print(i, "windows", ws.count, "nodes", n)
  if n > 200 { break }
  Thread.sleep(forTimeInterval: 1)
}
