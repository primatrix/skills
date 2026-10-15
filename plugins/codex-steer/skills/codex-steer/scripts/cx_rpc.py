"""Minimal JSON-RPC client for a private `codex app-server` (stdio). Shares ~/.codex thread store.

usage: cx_rpc.py '[["method", {params}], ...]'
Works for store-level calls such as thread/queue/list, thread/queue/delete. It CANNOT steer a turn
running inside the desktop app (that thread lives in the app's own server -> "thread not found").
"""
import json,subprocess,sys,select,time
C='/Applications/ChatGPT.app/Contents/Resources/codex-cli/CodexCLI.app/Contents/MacOS/codex'
p=subprocess.Popen([C,'app-server'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,bufsize=1)
def send(o): p.stdin.write(json.dumps(o)+'\n'); p.stdin.flush()
def recv(i,t=20):
    e=time.time()+t
    while time.time()<e:
        r,_,_=select.select([p.stdout],[],[],0.5)
        if not r: continue
        l=p.stdout.readline()
        if not l: return None
        try: m=json.loads(l)
        except: continue
        if m.get('id')==i: return m
send({'jsonrpc':'2.0','id':1,'method':'initialize','params':{'clientInfo':{'name':'claude-reviewer','version':'1'},'capabilities':{'experimentalApi':True}}})
print('init',json.dumps(recv(1))[:150])
send({'jsonrpc':'2.0','method':'initialized'})
for i,(m,pr) in enumerate(json.loads(sys.argv[1]),start=2):
    send({'jsonrpc':'2.0','id':i,'method':m,'params':pr}); print(m,json.dumps(recv(i))[:700])
p.terminate()
