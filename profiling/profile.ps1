python -m cProfile -o ./profiling/run_py.profile -s cumtime ./run.py
Write-Output 'sort cumtime
stats'|
python -m pstats .\profiling\run_py.profile > profiling/run_py_stats_dump.txt