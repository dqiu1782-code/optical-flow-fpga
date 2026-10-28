cd E:\hetero\code
$py="E:\lkenv\Scripts\python.exe"
$cm="E:\hetero\colmap_env\Library\bin\colmap.exe"
foreach ($k in 200,400,800) { $w="E:\hetero\work\fr3_glo_K$k"; New-Item -ItemType Directory -Force $w | Out-Null; Copy-Item E:\hetero\work\fr3\flow.npz,E:\hetero\work\fr3\flow_meta.json $w; & $py hetero_prep.py run --work $w --K $k --colmap $cm --methods B1 B2 O1 O2 O2@1.5 --fresh --mapper global; & $py hetero_prep.py report --work $w }
echo GLOBAL_DONE
foreach ($k in 100,400) { & $py run_3dgs.py --work E:\hetero\work\fr3_K$k --gs E:\hetero\code\gs --iters 30000 --methods B1 B2 O2; & $py hetero_prep.py report --work E:\hetero\work\fr3_K$k }
echo GS2_DONE
