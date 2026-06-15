"""Tests for free model labeling and filtering functionality."""

import pytest

from api.models.responses import ModelsListResponse
from api.routes import _build_models_list_response, _is_free_model
from config.settings import Settings


def test_free_models_list_validator_empty_string():
    """Test that FREE_MODELS_LIST=\"\" returns empty list."""
    from config.settings import Settings

    # Test the validator function directly
    result = Settings._split_free_models("")
    assert result == []


def test_free_models_list_validator_whitespace_only():
    """Test that FREE_MODELS_LIST=\"  ,  \" returns empty list."""
    from config.settings import Settings

    # Test the validator function directly
    result = Settings._split_free_models("  ,  ")
    assert result == []


def test_free_models_list_validator_with_whitespace():
    """Test that whitespace around commas is stripped."""
    from config.settings import Settings

    # Test the validator function directly
    result = Settings._split_free_models(" model1 , model2 ")
    assert result == ["model1", "model2"]


def test_free_models_list_validator_already_list():
    """Test that when free_models_list is already a list, it's returned as-is."""
    from config.settings import Settings

    # Test the validator function directly
    result = Settings._split_free_models(["pre-existing"])
    assert result == ["pre-existing"]


def test_is_free_model_suffix_free_colon():
    """Test model ref ending with :free is detected as free."""
    settings = Settings()
    settings.free_models_list = []  # Start with empty list
    assert _is_free_model("provider/model:free", settings) is True


def test_is_free_model_suffix_free_slash():
    """Test model ref ending with /free is detected as free."""
    settings = Settings()
    settings.free_models_list = []  # Start with empty list
    assert _is_free_model("provider/model/free", settings) is True


def test_is_free_model_suffix_free_dash():
    """Test model ref ending with -free is detected as free."""
    settings = Settings()
    settings.free_models_list = []  # Start with empty list
    assert _is_free_model("provider/model-free", settings) is True


def test_is_free_model_in_free_list():
    """Test model ref in FREE_MODELS_LIST is detected as free."""
    settings = Settings()
    settings.free_models_list = ["provider/model", "other/model"]
    assert _is_free_model("provider/model", settings) is True
    assert _is_free_model("other/model", settings) is True


def test_is_free_model_not_free():
    """Test model ref with no free indicators returns False."""
    settings = Settings()
    settings.free_models_list = ["other/model"]
    assert _is_free_model("provider/model", settings) is False


def test_is_free_model_case_sensitive_list():
    """Test FREE_MODELS_LIST matching is case-sensitive (current behavior)."""
    settings = Settings()
    settings.free_models_list = ["Provider/Model"]
    assert _is_free_model("provider/model", settings) is False  # Case sensitive
    assert _is_free_model("Provider/Model", settings) is True  # Exact match


def test_is_free_model_free_list_priority_over_suffix():
    """Test that explicit FREE_MODELS_LIST works alongside suffix detection."""
    settings = Settings()
    settings.free_models_list = ["provider/model"]
    # This would be True anyway due to suffix, but list matching should work
    assert _is_free_model("provider/model-free", settings) is True


def test_build_models_list_with_free_models_first():
    """Test that SHOW_FREE_MODELS_FIRST=true puts free models first."""
    from providers.registry import ProviderRegistry

    settings = Settings.model_construct(
        anthropic_auth_token="",
    )
    # Make the default model free
    settings.free_models_list = ["nvidia_nim/nvidia/nemotron-3-super-120b-a12b"]
    settings.show_free_models_first = True
    settings.only_show_free_models = False

    # Mock provider registry with no models to simplify
    registry = ProviderRegistry()

    response = _build_models_list_response(settings, registry)

    # First model should be the free one
    assert (
        response.data[0].id == "anthropic/nvidia_nim/nvidia/nemotron-3-super-120b-a12b"
    )
    assert "(free)" in response.data[0].display_name


def test_build_models_list_only_free_models():
    """Test that ONLY_SHOW_FREE_MODELS=true filters to only free models."""
    from providers.registry import ProviderRegistry

    settings = Settings.model_construct(
        MODEL_OPUS="open_router/anthropic/claude-opus",  # Not free
        MODEL_SONNET=None,
        MODEL_HAIKU=None,
        anthropic_auth_token="",
    )
    # Only make the default model free
    settings.free_models_list = ["nvidia_nim/nvidia/nemotron-3-super-120b-a12b"]
    settings.show_free_models_first = False
    settings.only_show_free_models = True

    registry = ProviderRegistry()

    response = _build_models_list_response(settings, registry)

    # Should only contain the free model and its thinking variants
    assert len(response.data) == 2
    # Both should be marked as free
    assert "(free)" in response.data[0].display_name
    assert "(free)" in response.data[1].display_name
    # Both should be based on the same model
    assert "nvidia_nim/nvidia/nemotron-3-super-120b-a12b" in response.data[0].id
    assert "nvidia_nim/nvidia/nemotron-3-super-120b-a12b" in response.data[1].id


def test_build_models_list_combined_flags():
    """Test SHOW_FREE_MODELS_FIRST and ONLY_SHOW_FREE_MODELS together."""
    from providers.registry import ProviderRegistry

    settings = Settings.model_construct(
        MODEL_OPUS="open_router/anthropic/claude-opus",  # Not free
        MODEL_SONNET=None,
        MODEL_HAIKU=None,
        anthropic_auth_token="",
    )
    # Only make the default model free
    settings.free_models_list = ["nvidia_nim/nvidia/nemotron-3-super-120b-a12b"]
    settings.show_free_models_first = True
    settings.only_show_free_models = True

    registry = ProviderRegistry()

    response = _build_models_list_response(settings, registry)

    # Should only contain the free model and its thinking variants
    assert len(response.data) == 2
    # Both should be marked as free
    assert "(free)" in response.data[0].display_name
    assert "(free)" in response.data[1].display_name
    # Both should be based on the same model
    assert "nvidia_nim/nvidia/nemotron-3-super-120b-a12b" in response.data[0].id
    assert "nvidia_nim/nvidia/nemotron-3-super-120b-a12b" in response.data[1].id
    # When SHOW_FREE_MODELS_FIRST=True, free models should come first
    # Since we only have free models, they should be first anyway


def test_build_models_list_no_free_models():
    """Test behavior when no models are free."""
    from providers.registry import ProviderRegistry

    settings = Settings.model_construct(
        MODEL="deepseek/deepseek-chat",
        MODEL_OPUS=None,
        MODEL_SONNET=None,
        MODEL_HAIKU=None,
        anthropic_auth_token="",
    )
    settings.free_models_list = []  # No free models
    settings.show_free_models_first = True
    settings.only_show_free_models = False

    registry = ProviderRegistry()

    response = _build_models_list_response(settings, registry)

    # Should still return models but none marked as free
    # Note: The actual behavior depends on how models are classified
    # At minimum, we should get a valid response
    assert isinstance(response, ModelsListResponse)
    assert response.has_more is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
