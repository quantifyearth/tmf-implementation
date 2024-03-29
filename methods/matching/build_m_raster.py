import argparse
import glob
import os
import shutil
import sys
import tempfile
import time
from multiprocessing import Manager, Process, Queue, cpu_count

from osgeo import gdal  # type: ignore

from yirgacheffe.layers import RasterLayer  # type: ignore


# We do not re-use data in this, so set a small block cache size for GDAL, otherwise
# it pointlessly hogs memory, and then spends a long time tidying it up after.
gdal.SetCacheMax(1024 * 1024 * 16)


def worker(
    filename: str,
    result_dir: str,
    compress: bool,
    input_queue: Queue,
) -> None:
    output_tif = os.path.join(result_dir, filename)

    calc = None
    final = None

    while True:
        path = input_queue.get()
        if path is None:
            break
        partial_raster = RasterLayer.layer_from_file(path)

        if final is None:
            final = RasterLayer.empty_raster_layer_like(partial_raster, filename=output_tif, compress=compress)

        if calc is None:
            calc = partial_raster
        else:
            calc = calc + partial_raster

    if final is not None:
        assert calc is not None
        calc.save(final)
        del final

def build_k(
    images_dir: str,
    output_filename: str,
    processes_count: int
) -> None:

    files = [os.path.join(images_dir, x) for x in glob.glob("*.tif", root_dir=images_dir)]

    with tempfile.TemporaryDirectory() as tempdir:
        with Manager() as manager:
            source_queue = manager.Queue()

            workers = [Process(target=worker, args=(
                f"{index}.tif",
                tempdir,
                False,
                source_queue
            )) for index in range(processes_count)]
            for worker_process in workers:
                worker_process.start()

            for file in files:
                source_queue.put(file)
            for _ in range(len(workers)):
                source_queue.put(None)

            processes = workers
            while processes:
                candidates = [x for x in processes if not x.is_alive()]
                for candidate in candidates:
                    candidate.join()
                    if candidate.exitcode:
                        for victim in processes:
                            victim.kill()
                        sys.exit(candidate.exitcode)
                    processes.remove(candidate)
                time.sleep(1)

            # here we should have now a set of images in tempdir to merge
            result_dir, filename = os.path.split(output_filename)
            single_worker = Process(target=worker, args=(
                filename,
                tempdir,
                True,
                source_queue
            ))
            single_worker.start()
            nextfiles = [os.path.join(tempdir, x) for x in glob.glob("*.tif", root_dir=tempdir)]
            for file in nextfiles:
                source_queue.put(file)
            source_queue.put(None)

            processes = [single_worker]
            while processes:
                candidates = [x for x in processes if not x.is_alive()]
                for candidate in candidates:
                    candidate.join()
                    if candidate.exitcode:
                        for victim in processes:
                            victim.kill()
                        sys.exit(candidate.exitcode)
                    processes.remove(candidate)
                time.sleep(1)

            shutil.move(os.path.join(tempdir, filename), os.path.join(result_dir, filename))

def main() -> None:
    parser = argparse.ArgumentParser(description="Generates a single raster of M from all the per K rasters.")
    parser.add_argument(
        "--rasters_directory",
        type=str,
        required=True,
        dest="rasters_directory",
        help="Directory containing rasters generated by find_potential_matches.py"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        dest="output_filename",
        help="Destination GeoTIFF file for results."
    )
    parser.add_argument(
        "-j",
        type=int,
        required=False,
        default=round(cpu_count() / 2),
        dest="processes_count",
        help="Number of concurrent threads to use."
    )
    args = parser.parse_args()

    build_k(
        args.rasters_directory,
        args.output_filename,
        args.processes_count
    )

if __name__ == "__main__":
    main()
