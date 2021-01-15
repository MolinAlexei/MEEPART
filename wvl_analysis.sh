python3 analyze.py --wvl 2 --resolution 7 --file_name '2mm'
python3 analyze.py --plot_FT --file_name '2mm' --wvl 2
python3 analyze.py --wvl 5 --resolution 4 --file_name '5mm'
python3 analyze.py --plot_FT --file_name '5mm' --wvl 5
python3 analyze.py --wvl 7 --resolution 2 --file_name '7mm'
python3 analyze.py --plot_FT --file_name '7mm' --wvl 7
python3 analyze.py --wvl 10 --resolution 2 --file_name '10mm'
python3 analyze.py --plot_FT --file_name '2mm' --wvl 10

python3 analyze.py --wvl 2 --resolution 7 --file_name '2mm_multibeam' --beam_nb 3
python3 analyze.py --plot_FT --file_name '2mm_multibeam' --wvl 2
python3 analyze.py --wvl 5 --resolution 4 --file_name '5mm_multibeam'  --beam_nb 3
python3 analyze.py --plot_FT --file_name '5mm_multibeam' --wvl 5
python3 analyze.py --wvl 7 --resolution 2 --file_name '7mm_multibeam'  --beam_nb 3
python3 analyze.py --plot_FT --file_name '7mm_multibeam' --wvl 7
python3 analyze.py --wvl 10 --resolution 2 --file_name '10mm_multibeam'  --beam_nb 3 
python3 analyze.py --plot_FT --file_name '10mm_multibeam' --wvl 10