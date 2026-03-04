import xarray as xr
import numpy as np
from next_tests.integration_tests.multi_feature_tests.fvm_nabla_setup import nabla_setup
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import exec_alloc_descriptor


ds = xr.open_dataset(
    "/home/raphael/Documents/Studium/Msc_thesis/grid-generator/parallelogram_grid.nc"
)

print(ds.attrs)
print(ds.coords)
print(ds.data_vars)
# E2V: (nc, edge) -> (edge, nc)
e2v = ds["edge_vertices"].transpose("edge", "nc").values.astype(np.int32)
e2v = np.where(e2v > 0, e2v - 1, -1)

# V2E: (ne, vertex) -> (vertex, ne)
v2e = ds["edges_of_vertex"].transpose("vertex", "ne").values.astype(np.int32)
v2e = np.where(v2e > 0, v2e - 1, -1)

lon = ds["longitude_vertices"].values.astype(np.float64)
lat = ds["latitude_vertices"].values.astype(np.float64)
lonlat = np.stack([lon, lat], axis=1)  # (n_vertex, 2), degrees

dual_volumes = ds["dual_area"].values.astype(np.float64)

dual_normals = np.stack(
    [
        ds["zonal_normal_dual_edge"].values.astype(np.float64),
        ds["meridional_normal_dual_edge"].values.astype(np.float64),
    ],
    axis=1,
)

def test_build_setup_from_nc(exec_alloc_descriptor):
    setup = nabla_setup.from_connectivity(
        allocator=exec_alloc_descriptor.allocator,
        e2v=e2v,
        v2e=v2e,
        lonlat_deg=lonlat,
        dual_normals=dual_normals,
        dual_volumes=dual_volumes,
    )
    assert setup.nodes_size > 0