# Tarzan

Tar, as a high performance streamable format, has been widely used in the DL community
(e.g. [TorchData](https://github.com/pytorch/data), [WebDataset](https://github.com/webdataset/webdataset)).
[TFDS](https://www.tensorflow.org/datasets/add_dataset)-like dataset builder API provides a high-level interface for
users to build their own datasets, and is also adopted
by [HuggingFace](https://huggingface.co/docs/datasets/main/en/image_dataset#loading-script).

Why not connect the two? Tarzan provides a minimal high-level API to help users build their own Tar-based datasets. It
also maps well between nested feature and Tar file structure to let you peek into the Tar file without extracting it.

## Installation

```bash
pip install tarzan
```

## Quick Start

1. Define your dataset info, which describes the dataset structure and any metadata.
```python
from tarzan.info import DatasetInfo
from tarzan.features import Features, Text, Scalar, Tensor, Audio

info = DatasetInfo(
   description="A fake dataset",
   features=Features({
       'single': Text(),
       'nested_list': [Scalar('int32')],
       'nested_dict': {
           'inner': Tensor(shape=(None, 3), dtype='float32'),
       },
       'complex': [{
           'inner_1': Text(),
           'inner_2': Audio(sample_rate=16000),
       }]
   }),
   metadata={
       'version': '1.0.0'
   }
)
```

2. Write your data to Tar files with `ShardWriter`.
```python
from tarzan.writers import ShardWriter 
with ShardWriter('data_dir', info, max_count=2) as writer:
   for i in range(5):
      writer.write({
          'single': 'hello',
          'nested_list': [1, 2, 3],
          'nested_dict': {
              'inner': [[1, 2, 3], [4, 5, 6]]
          },
          'complex': [{
              'inner_1': 'world',
              'inner_2': 'audio.wav'
          }]
      })
```
The structure of the `data_dir` is as follows:
```text
data_dir
├── 00000.tar
├── 00001.tar
├── 00002.tar
└── dataset_info.json
```
`max_count` and `max_size` control the maximum number of samples and the maximum size of each shard. Here we set the
`max_count` to 2 to create 3 shards.
`dataset_info.json` is a json file serialized from `info, which we rely on to read the data later.
```bash
cat data_dir/dataset_info.json
```
```json
{
  "description": "A fake dataset",
  "file_list": [
    "00000.tar",
    "00000.tar",
    "00001.tar",
    "00002.tar"
  ],
  "features": {
    "single": {
      "_type": "Text"
    },
    "nested_list": [
      {
        "shape": [],
        "dtype": "int32",
        "_type": "Scalar"
      }
    ],
    "nested_dict": {
      "inner": {
        "shape": [
          null,
          3
        ],
        "dtype": "float32",
        "_type": "Tensor"
      }
    },
    "complex": [
      {
        "inner_1": {
          "_type": "Text"
        },
        "inner_2": {
          "shape": [
            null
          ],
          "dtype": "float32",
          "_type": "Audio",
          "sample_rate": 16000
        }
      }
    ]
  },
  "metadata": {
    "version": "1.0.0"
  }
}
```
You can peek the tar file without extracting it and it should map well to the nested feature structure.
```bash
tree data_dir/00000.tar
```
```text
.
├── 0
│   ├── complex
│   │   └── 0
│   │       ├── inner_1
│   │       └── inner_2
│   ├── nested_dict
│   │   └── inner
│   ├── nested_list
│   │   ├── 0
│   │   ├── 1
│   │   └── 2
│   └── single
└── 1
    ├── complex
    │   └── 0
    │       ├── inner_1
    │       └── inner_2
    ├── nested_dict
    │   └── inner
    ├── nested_list
    │   ├── 0
    │   ├── 1
    │   └── 2
    └── single
```
3.Read the dataset with `TarReader`
```python
from tarzan.readers import TarReader
reader = TarReader.from_dataset_info('data_dir/dataset_info.json')

for tar_name, idx, example in reader:
    print(tar_name, idx, example)
```
```text
data_dir/00000.tar 0 {'nested_dict': {'inner': array([[1., 2., 3.],
       [4., 5., 6.]], dtype=float32)}, 'single': 'hello', 'complex': [{'inner_1': 'world', 'inner_2': <tarzan.features.audio.AudioDecoder object at 0x7fb8903443d0>}], 'nested_list': [array(1, dtype=int32), array(2, dtype=int32), array(3, dtype=int32)]}
...
```
Note that the `Audio` feature is returned as a lazy read object `AudioDecoder` to avoid unnecessary read for large audio.
