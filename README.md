(from root directory)

```bash
cd data

for file in *.gz; do gunzip "$file"; done
```