from pipeline.logging_utils import format_block_progress, should_use_timestep_progress


def test_format_block_progress_uses_requested_block_and_frame_counts():
    assert format_block_progress(
        block_index=3,
        total_blocks=8,
        block_start_frame=9,
        total_frames=33,
    ) == "block 3/8 - 9/33"


def test_should_use_timestep_progress_requires_main_process_and_tty():
    assert should_use_timestep_progress(
        main_process=True,
        stdout_isatty=True,
        stderr_isatty=False,
    ) is True
    assert should_use_timestep_progress(
        main_process=True,
        stdout_isatty=False,
        stderr_isatty=False,
    ) is False
    assert should_use_timestep_progress(
        main_process=False,
        stdout_isatty=True,
        stderr_isatty=True,
    ) is False
