"""Compute brain region statistics across all downloaded sessions (Beryl mapping)."""
import sys
from pathlib import Path
import numpy as np
import datasets
from iblatlas.regions import BrainRegions

BASE_PATH = "/media/lenny-aharon/T7/ibl-mouse/ibl-mouse_neural-activity"
DATA_DIR = Path(BASE_PATH) / "ibl_aligned"

# The 39 sessions used for decoding analysis
eids = [
    "15b69921-d471-4ded-8814-2adad954bcd8",
    "15763234-d21e-491f-a01b-1238eb96d389",
    "aad23144-0e52-4eac-80c5-c4ee2decb198",
    "9b528ad0-4599-4a55-9148-96cc1d93fb24",
    "4b00df29-3769-43be-bb40-128b1cba6d35",
    "f312aaec-3b6f-44b3-86b4-3a0c119c0438",
    "0f77ca5d-73c2-45bd-aa4c-4c5ed275dbde",
    "1b715600-0cbc-442c-bd00-5b0ac2865de1",
    "3e6a97d3-3991-49e2-b346-6948cb4580fb",
    "3f859b5c-e73a-4044-b49e-34bb81e96715",
    "4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b",
    "5ae68c54-2897-4d3a-8120-426150704385",
    "5dcee0eb-b34d-4652-acc3-d10afc6eae68",
    "7cb81727-2097-4b52-b480-c89867b5b34c",
    "8a3a0197-b40a-449f-be55-c00b23253bbf",
    "30c4e2ab-dffc-499d-aae4-e51d6b3218c2",
    "51e53aff-1d5d-4182-a684-aba783d50ae5",
    "746d1902-fa59-4cab-b0aa-013be36060d5",
    "824cf03d-4012-4ab1-b499-c83a92c5589e",
    "3638d102-e8b6-4230-8742-e548cd87a949",
    "6899a67d-2e53-4215-a52a-c7021b5da5d4",
    "8928f98a-b411-497e-aa4b-aa752434686d",
    "46794e05-3f6a-4d35-afb3-9165091a5a74",
    "493170a6-fd94-4ee4-884f-cc018c17eeb9",
    "54238fd6-d2d0-4408-b1a9-d19d24fd29ce",
    "73918ae1-e4fd-4c18-b132-00cb555b1ad2",
    "b03fbc44-3d8e-4a6c-8a50-5ea3498568e0",
    "b22f694e-4a34-4142-ab9d-2556c3487086",
    "c7248e09-8c0d-40f2-9eb4-700a8973d8c8",
    "c7bf2d49-4937-4597-b307-9f39cb1c7b16",
    "ca4ecb4c-4b60-4723-9b9e-2c54a6290a53",
    "d0ea3148-948d-4817-94f8-dcaf2342bbbe",
    "d23a44ef-1402-4ed7-97f5-47e9a7a504d9",
    "dc962048-89bb-4e6a-96a9-b062a2be1426",
    "e535fb62-e245-4a48-b119-88ce62a6fe67",
    "f115196e-8dfe-4d2a-8af3-8206d93c1729",
    "f140a2ec-fd49-4814-994a-fe3476f14e66",
    "f3ce3197-d534-4618-bf81-b687555d1883",
    "ff96bfe1-d925-4553-94b5-bf8297adf259",
]
print(f"Found {len(eids)} sessions\n")

brainreg = BrainRegions()

all_regions_global = set()
regions_per_session = {}

for i, eid in enumerate(eids):
    ds = datasets.load_from_disk(str(DATA_DIR / eid))
    cluster_regions = np.array(ds["train"]["cluster_regions"])[0]
    # Apply Beryl mapping (same as list_brain_regions in ibl_data_utils.py)
    beryl_regions = brainreg.acronym2acronym(cluster_regions, mapping="Beryl")
    unique = set(beryl_regions)
    regions_per_session[eid] = unique
    all_regions_global.update(unique)
    print(f"[{i+1:2d}/{len(eids)}] {eid}: {len(unique):3d} regions, {len(cluster_regions):4d} neurons")

counts = [len(r) for r in regions_per_session.values()]

print("\n" + "=" * 60)
print(f"Total sessions:              {len(eids)}")
print(f"Total unique brain regions:  {len(all_regions_global)}")
print(f"Regions per session  min:    {np.min(counts)}")
print(f"Regions per session  median: {np.median(counts):.1f}")
print(f"Regions per session  max:    {np.max(counts)}")
print(f"Regions per session  mean:   {np.mean(counts):.1f}")
print("=" * 60)

print("\nAll unique regions (sorted):")
for r in sorted(all_regions_global):
    print(f"  {r}")
