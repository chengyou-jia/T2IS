

## Usage

### Basic Usage
```bash
python inference_t2is.py
```

### With Parameters
```bash
python inference_t2is.py --idx 0 --json_file ./data/gallery.json --output_dir ./output_images/
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--idx` | int | None | Parameter index from JSON file |
| `--json_file` | str | `./data/gallery.json` | Input JSON file path |
| `--output_dir` | str | `./output_images/` | Output directory path |

## Output Structure

Generated images are saved in:
```
output_images/
└── seed_*/
    └── {idx}_merge_seed*.png
```

## Generated ImageSet
<summary>Examples</summary> 
<table class="center">
  <tr>
    <td width=100% style="border: none"><img src="pic/0001_0007_merge_seed1234.png" style="width:100%"></td>
  </tr>
  <td width="25%" style="border: none; text-align: center; word-wrap: break-word">
      <pre style="background-color: #f4f4f4; padding: 10px; border-radius: 5px; font-family: Consolas, monospace; font-size: 16px; display: inline-block;">python inference_t2is.py --idx=</pre>
  </td>
</table>