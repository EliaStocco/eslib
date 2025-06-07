/usr/lpp/mmfs/bin/mmlsquota

find . -xdev -type f | cut -d/ -f2 | sort | uniq -c | sort -rn | head -20
conda clean --all --yes
/usr/lpp/mmfs/bin/mmlsquota

rm -rf ~/.cache/*
rm -rf ~/.vscode-server/*
find ~ -type d -name "__pycache__" -exec rm -rf {} +
/usr/lpp/mmfs/bin/mmlsquota

du -h ~/.local --max-depth=1 | sort -hr
du -h ~/.local/lib --max-depth=1 | sort -hr
rm -rf ~/.local/lib/python3.9
rm -rf ~/.local/lib/python3.10
rm -rf ~/.local/bin/*

/usr/lpp/mmfs/bin/mmlsquota | awk '
/^ada_u/ {
  disk_used=$4
  disk_quota=$5
  file_used=$10
  file_quota=$11

  disk_pct = (disk_used / disk_quota) * 100
  file_pct = (file_used / file_quota) * 100

  printf "Disk usage: %.2f%%\nFile usage: %.2f%%\n", disk_pct, file_pct
}'

