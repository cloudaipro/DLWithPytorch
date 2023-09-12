from dsets import LunaDataset, getCt
from torch.utils.data import DataLoader

lunaDataset = LunaDataset()
# loader = DataLoader(
#     lunaDataset,
#     batch_size=64,
#     num_workers=8,
#     pin_memory=False
# )

# totalCount = len(loader)
# for idx, candidate_t, pos_t, series_uid, center_irc in enumerate(loader):
#     progress = float(float(idx) * 100.0/float(totalCount))
#     print(f">> {progress:.1f}%, {idx}/{totalCount}", end='\r', flush=True)

count = 0
ngCount = 0
totalCount = len(lunaDataset.candidateInfo_list)
for ndx, info in enumerate(lunaDataset.candidateInfo_list):
    ct = getCt(info.series_uid)
    if ct.origin_xyz == None:
        ngCount += 1
        continue
    candidate_t, pos_t, series_uid, center_irc = lunaDataset[ndx]
    if center_irc != None:
        count += 1
    else:
        print(f'\nError: series_uid={info.series_uid}, ndx={ndx}')
        # candidate_t, pos_t, series_uid, center_irc = lunaDataset.getitemV2(ndx)
        # count += 1
    progress = float(float(count) * 100.0/float(totalCount))
    print(f">> {progress:.1f}%, {count}/{totalCount}", end='\r', flush=True)

print(f"\nDone!")