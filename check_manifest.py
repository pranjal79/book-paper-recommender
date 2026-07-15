import json

with open("pipeline_manifest.json") as f:
    m = json.load(f)

print(f"Run name : {m['run_name']}")
print(f"Status   : {m['status']}")
print(f"Started  : {m['start_time']}")
print(f"Ended    : {m['end_time']}")

print("\nStage breakdown:")
for stage, info in m["stages"].items():
    print(f"  {stage:<25} → {info['status']} ({info.get('elapsed_s', 0):.1f}s)")