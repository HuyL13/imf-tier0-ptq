from imf_tier0.gpu.runtime import release_cuda, select_batch_size


class FakeCuda:
    def __init__(self):
        self.emptied = 0
        self.reset = 0

    def is_available(self):
        return True

    def empty_cache(self):
        self.emptied += 1

    def reset_peak_memory_stats(self):
        self.reset += 1


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()


def test_batch_probe_selects_largest_successful_candidate() -> None:
    seen = []

    def probe(batch_size: int) -> bool:
        seen.append(batch_size)
        return batch_size <= 4

    assert select_batch_size(probe, [1, 2, 4, 8]) == 4
    assert seen == [8, 4]


def test_release_cuda_clears_allocator_when_available() -> None:
    torch = FakeTorch()
    release_cuda(torch_module=torch)
    assert torch.cuda.emptied == 1

