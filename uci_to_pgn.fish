#! /usr/bin/env nix-shell
#! nix-shell -i fish -p fish
python py_uci_to_pgn.py $argv | tail -n 1 | wl-copy
