import io
import pathlib
from src.utils import WorldDefinition


def test_read_pgw_file(tmp_path: pathlib.Path):

    data = """0.0311407714
    -0.0030169731
    -0.0032656759
    -0.0310992985
    619245.1776260373
    5809189.9721523859
    """

    pgw_file = tmp_path / "test.pgw"
    pgw_file.write_text(data)

    world = WorldDefinition.from_pgw_file(pgw_file)

    extent = world.get_image_extent(3840, 2160)

    width_meters = extent[1] - extent[0]
    assert width_meters > 20

    height_meters = extent[3] - extent[2]
    assert height_meters > 20
