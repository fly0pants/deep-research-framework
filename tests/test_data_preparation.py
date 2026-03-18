import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.engine.data_preparation import DataPreparator


@pytest.fixture
def mock_openai():
    client = MagicMock()
    client.vector_stores.create.return_value = MagicMock(id="vs_test123")
    client.files.create.return_value = MagicMock(id="file_test123")
    client.vector_stores.files.create.return_value = MagicMock(id="vsf_test123")
    client.vector_stores.delete.return_value = None
    return client


@pytest.fixture
def preparator(mock_openai, tmp_path):
    return DataPreparator(openai_client=mock_openai, temp_dir=tmp_path)


def _make_mock_http_client(get_side_effect=None, get_return_value=None):
    """Create a mock AsyncClient that works as an async context manager."""
    mock_client = AsyncMock()
    if get_side_effect:
        mock_client.get.side_effect = get_side_effect
    elif get_return_value:
        mock_client.get.return_value = get_return_value
    mock_cls = MagicMock()
    mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
    return mock_cls


async def test_prefetch_api_data(preparator):
    api_config = {
        "name": "test_api",
        "base_url": "https://api.test.com/v1",
        "auth": {"type": "bearer", "token_env": "TEST_KEY"},
        "prefetch": [{"endpoint": "/data", "params": {"limit": 10}}],
    }
    mock_resp = MagicMock(
        status_code=200,
        json=lambda: {"items": [1, 2, 3]},
        text='{"items": [1, 2, 3]}',
    )
    mock_resp.raise_for_status = MagicMock()
    mock_cls = _make_mock_http_client(get_return_value=mock_resp)

    with patch.dict("os.environ", {"TEST_KEY": "fake-key"}):
        with patch("httpx.AsyncClient", mock_cls):
            results = await preparator.prefetch(api_config)
    assert len(results) == 1
    assert results[0]["endpoint"] == "/data"
    assert results[0]["data"] == {"items": [1, 2, 3]}


async def test_prefetch_handles_failure(preparator):
    api_config = {
        "name": "test_api",
        "base_url": "https://api.test.com/v1",
        "auth": {"type": "bearer", "token_env": "TEST_KEY"},
        "prefetch": [{"endpoint": "/data", "params": {}}],
    }
    mock_cls = _make_mock_http_client(get_side_effect=Exception("connection error"))

    with patch.dict("os.environ", {"TEST_KEY": "fake-key"}):
        with patch("httpx.AsyncClient", mock_cls):
            results = await preparator.prefetch(api_config)
    assert len(results) == 0


async def test_create_vector_store(preparator, mock_openai):
    prefetch_results = [
        {"endpoint": "/data", "data": {"items": [1, 2, 3]}},
    ]
    api_docs_content = "openapi: 3.0.0"
    vs_id = await preparator.create_vector_store(
        task_id="dr_test",
        prefetch_results=prefetch_results,
        api_docs_content=api_docs_content,
    )
    assert vs_id == "vs_test123"
    assert mock_openai.vector_stores.create.called


async def test_cleanup_vector_store(preparator, mock_openai):
    await preparator.cleanup("vs_test123")
    mock_openai.vector_stores.delete.assert_called_once_with("vs_test123")
