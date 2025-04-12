# cutaway
[Hey Peter, remember that time...](https://tvtropes.org/pmwiki/pmwiki.php/CutawayGag/FamilyGuy)

![Iraq Lobster](.assets/iraq_lobster.gif)

I have a friend who stores a good amount of video of us hanging out online. The goal of this project is to create a simple video search engine to get clips from a text search. This project will rely on `ImageBind`.
> The model learns a single embedding, or shared representation space, not just for text, image/video, and audio, but also for sensors that record depth (3D), thermal (infrared radiation), and inertial measurement units (IMU), which calculate motion and position.

## Requirements
- Python 3.10 x86_64

For MacOS:
Some dependencies aren't ARM ready. Use Rosetta 2.
```sh
softwareupdate --install-rosetta
arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
arch -x86_64 brew install python@3.10
uv venv --python /usr/local/opt/python@3.10/bin/python3.10
uv run main.py
```

## Citations
- [Creating an In-Video Search System](https://suyashthakurblog.hashnode.dev/creating-an-in-video-search-system)

- [ImageBind Repo](https://github.com/facebookresearch/ImageBind)
```
@inproceedings{girdhar2023imagebind,
  title={ImageBind: One Embedding Space To Bind Them All},
  author={Girdhar, Rohit and El-Nouby, Alaaeldin and Liu, Zhuang
and Singh, Mannat and Alwala, Kalyan Vasudev and Joulin, Armand and Misra, Ishan},
  booktitle={CVPR},
  year={2023}
}
```
