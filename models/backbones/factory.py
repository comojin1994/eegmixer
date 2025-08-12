from huggingface_hub import PyTorchModelHubMixin, hf_hub_download, constants
from huggingface_hub.errors import EntryNotFoundError
from typing import Dict, List, Optional, Union
from pathlib import Path
from utils.setup_utils import get_huggingface_token
import os


class FactoryModelHubMixin(PyTorchModelHubMixin):
    def __init_subclass__(
        cls, *args, tags: Optional[List[str]] = None, **kwargs
    ) -> None:
        tags = tags or []
        tags.append("pytorch_model_hub_mixin")
        kwargs["tags"] = tags
        cls.token = get_huggingface_token()
        return super().__init_subclass__(*args, **kwargs)

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        subfolder: Optional[str],
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: Optional[bool],
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs,
    ):
        """Load Pytorch pretrained weights and return the loaded model."""
        model = cls(**model_kwargs)
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, constants.SAFETENSORS_SINGLE_FILE)
            return cls._load_as_safetensor(model, model_file, map_location, strict)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    subfolder=subfolder,
                    filename=constants.SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=cls.token,
                    local_files_only=local_files_only,
                )
                return cls._load_as_safetensor(model, model_file, map_location, strict)
            except EntryNotFoundError:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    subfolder=subfolder,
                    filename=constants.PYTORCH_WEIGHTS_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=cls.token,
                    local_files_only=local_files_only,
                )
                return cls._load_as_pickle(model, model_file, map_location, strict)
