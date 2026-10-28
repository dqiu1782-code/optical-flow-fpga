cd E:\hetero\code
$py="E:\lkenv\Scripts\python.exe"
$cm="E:\hetero\colmap_env\Library\bin\colmap.exe"
$seq="E:\hetero\data\rgbd_dataset_freiburg3_long_office_household"
$w="E:\hetero\work\fr3"
if (!(Test-Path "$w\flow.npz")) { & $py hetero_prep.py flow --seq $seq --work $w --jobs 16 }
& $py hetero_prep.py run --work $w --K 200 --colmap $cm --methods B1 B2 O1 O2 O2@1.0 O2@1.5 --fresh
& $py hetero_prep.py report --work $w
echo PREP_DONE
