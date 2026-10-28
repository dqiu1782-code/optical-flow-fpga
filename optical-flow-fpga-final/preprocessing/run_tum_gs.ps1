cd E:\hetero\code
$py="E:\lkenv\Scripts\python.exe"
$w="E:\hetero\work\fr3"
& $py run_3dgs.py --work $w --gs E:\hetero\code\gs --iters 30000 --methods $args
& $py hetero_prep.py report --work $w
echo GS_DONE
