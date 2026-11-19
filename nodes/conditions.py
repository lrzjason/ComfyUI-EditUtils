"""Saved-condition cache nodes (models/conditions)."""

import torch
import os
import folder_paths

condition_dir = os.path.join(folder_paths.models_dir, "conditions")


if "conditions" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["conditions"] = ([condition_dir], {'.ckpt'})


class SaveCondition_EditUtils:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition": ("CONDITIONING",),
                "filename": ("STRING", {"default": "condition_tensor"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_condition"
    CATEGORY = "advanced/conditioning"
    OUTPUT_NODE = True

    def save_condition(self, condition, filename):
        os.makedirs(condition_dir, exist_ok=True)

        safe_filename = os.path.basename(filename)
        if not safe_filename.endswith(".ckpt"):
            safe_filename += ".ckpt"

        save_dir = folder_paths.folder_names_and_paths["conditions"][0][0]
        save_path = os.path.join(save_dir, safe_filename)

        try:
            tensors_to_save = []
            for cond, cond_info in condition:
                tensors_to_save.append((cond.clone(), cond_info))
            torch.save(tensors_to_save, save_path)
            print(f"Condition tensor saved to {save_path}")
        except Exception as e:
            print(f"Error saving condition tensor: {e}")

        return ()


class LoadCondition_EditUtils:
    @classmethod
    def INPUT_TYPES(cls):
        os.makedirs(condition_dir, exist_ok=True)

        condition_files = []
        try:
            condition_files = folder_paths.get_filename_list("conditions")
            condition_files = [os.path.splitext(file)[0] for file in condition_files]
        except Exception as e:
            print(f"Error listing condition files: {e}")

        if not condition_files:
            condition_files = ["No files found"]

        return {
            "required": {
                "filename": (condition_files, ),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "load_condition"
    CATEGORY = "advanced/conditioning"

    def load_condition(self, filename):
        os.makedirs(condition_dir, exist_ok=True)

        safe_filename = os.path.basename(filename)
        if not safe_filename.endswith(".ckpt"):
            safe_filename += ".ckpt"

        load_dir = folder_paths.folder_names_and_paths["conditions"][0][0]
        load_path = os.path.join(load_dir, safe_filename)

        try:
            if not os.path.exists(load_path):
                raise FileNotFoundError(f"Condition file not found: {load_path}")
            loaded_tensors = torch.load(load_path)
            print(f"Condition tensor loaded from {load_path}")
            return (loaded_tensors,)
        except Exception as e:
            print(f"Error loading condition tensor: {e}")
            return ([],)


class LoadConditionFromLoras_EditUtils:
    @classmethod
    def INPUT_TYPES(cls):
        os.makedirs(condition_dir, exist_ok=True)

        condition_files = []
        try:
            condition_files = folder_paths.get_filename_list("loras")
            condition_files = [os.path.splitext(file)[0] for file in condition_files]
        except Exception as e:
            print(f"Error listing condition files: {e}")

        if not condition_files:
            condition_files = ["No files found"]

        return {
            "required": {
                "filename": (condition_files, ),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "load_condition"
    CATEGORY = "advanced/conditioning"

    def load_condition(self, filename):
        os.makedirs(condition_dir, exist_ok=True)

        safe_filename = os.path.basename(filename)
        if not safe_filename.endswith(".ckpt"):
            safe_filename += ".ckpt"

        load_dir = folder_paths.folder_names_and_paths["conditions"][0][0]
        load_path = os.path.join(load_dir, safe_filename)

        try:
            if not os.path.exists(load_path):
                raise FileNotFoundError(f"Condition file not found: {load_path}")
            loaded_tensors = torch.load(load_path)
            print(f"Condition tensor loaded from {load_path}")
            return (loaded_tensors,)
        except Exception as e:
            print(f"Error loading condition tensor: {e}")
            return ([],)
