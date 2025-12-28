# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2025-12-27

### Added

- **Enhanced Gradio UI**: Professional interface with sidebar controls and inspection panels:
  - Sidebar with API key inputs, Bio-Lock domain selector, and real-time session statistics
  - Expandable accordion for viewing retrieved context and system logs
  - Live streaming updates for cost tracking and recursion depth
  - Gradio 6.2.0 compatibility fixes
- **Logging Infrastructure**: Created `src/utils/logger.py` with `GradioLogHandler` for UI log capture
- **UI Integration Tests**: Comprehensive test suite with proper Gradio mocking (`tests/test_ui_integration.py`)
- **Production-Grade Zep Tests**: Added retry logic and authentication validation tests (`tests/test_memory_manager_production.py`)
- **Comprehensive Unit Test Suite**: Generated 16 tests for `orchestrator.py` and `memory_manager.py`, achieving 94% statement coverage.
- **Project Documentation**: Created `README.md`, `API_DOCUMENTATION.md`, and `CHANGELOG.md`.
- **Google-Style Docstrings**: Added inline documentation to all core source files in `src/`.
- **Bio-Lock Filtering**: Implemented domain-specific context retrieval in `MemoryManager`.

### Changed

- **Dependency Overhaul**: Updated multiple packages for security and maintenance:
  - `aiohttp`: 3.11.11 -> 3.13.2 (Security fix)
  - `langchain-core`: 1.2.3 -> 1.2.5 (Security fix)
  - `openai`: 2.11.0 -> 2.14.0
  - `pytest`: 8.4.2 -> 9.0.2
- **Standardized Tests**: Migrated orchestrator tests to the `unittest` framework.

### Fixed

- **Zep Cloud Imports**: Corrected import path from `zep_cloud.api` to `zep_cloud.types` for Message and SearchFilters
- **Doc-Code Alignment**: Resolved discrepancies between implementation and documentation regarding node topology and streaming methods.
- **Mock Integrity**: Improved `MemoryManager` mock mode for robust local development.
