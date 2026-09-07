from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_defaults_match_documented_product_behaviour() -> None:
    settings = Settings()

    assert settings.default_scale == 4
    assert settings.default_model == "RealESRGAN_x4plus"
    assert settings.device == "auto"
    assert settings.use_fp16 is True


def test_upload_size_is_exposed_in_bytes() -> None:
    assert Settings(max_upload_size_mb=32).max_upload_size_bytes == 32 * 1024 * 1024


def test_csv_environment_values_are_split() -> None:
    settings = Settings(cors_origins="http://a.test, http://b.test")  # type: ignore[arg-type]

    assert settings.cors_origins == ["http://a.test", "http://b.test"]


def test_storage_subdirectories_derive_from_storage_dir(tmp_path: Path) -> None:
    settings = Settings(storage_dir=tmp_path)

    assert settings.inputs_dir == tmp_path / "inputs"
    assert settings.outputs_dir == tmp_path / "outputs"
    assert settings.thumbs_dir == tmp_path / "thumbs"


def test_ensure_directories_creates_the_storage_tree(tmp_path: Path) -> None:
    settings = Settings(storage_dir=tmp_path / "s", models_dir=tmp_path / "m")

    settings.ensure_directories()

    assert settings.inputs_dir.is_dir()
    assert settings.outputs_dir.is_dir()
    assert settings.thumbs_dir.is_dir()
    assert settings.models_dir.is_dir()


@pytest.mark.parametrize("invalid", [0, -1, 9999])
def test_upload_size_bounds_are_enforced(invalid: int) -> None:
    with pytest.raises(ValidationError):
        Settings(max_upload_size_mb=invalid)


def test_csv_lists_load_from_a_dotenv_file(tmp_path: Path) -> None:
    """Regression: pydantic-settings JSON-decodes complex types inside the
    dotenv source before field validators run. Without NoDecode, a plain
    comma-separated value in .env raises SettingsError at startup."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173\nALLOWED_FORMATS=JPEG,PNG\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

    assert settings.cors_origins == ["http://localhost:5173", "http://127.0.0.1:5173"]
    assert settings.allowed_formats == ["JPEG", "PNG"]


def test_json_lists_still_load_from_a_dotenv_file(tmp_path: Path) -> None:
    """The documented JSON form must keep working alongside the CSV form."""
    env_file = tmp_path / ".env"
    env_file.write_text('CORS_ORIGINS=["http://a.test","http://b.test"]\n', encoding="utf-8")

    settings = Settings(_env_file=env_file)  # type: ignore[call-arg]

    assert settings.cors_origins == ["http://a.test", "http://b.test"]
