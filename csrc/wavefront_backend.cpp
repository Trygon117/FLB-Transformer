#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// FORWARD PASS
void execute_forward(
    int num_ticks, int num_ports,
    py::function run_fetch_kernel, py::function batched_forward,
    torch::Tensor active_cells_buffer, torch::Tensor static_x,
    torch::Tensor static_stacked_grids, torch::Tensor routing_map,
    torch::Tensor port_map, torch::Tensor spatial_map_buffer,
    torch::Tensor gathered_out_buffer, py::dict bwd_cache,
    py::dict stacked_params, py::object config
) {
    for (int i = 0; i < num_ticks; ++i) {
        torch::Tensor cell_indices = active_cells_buffer[i];
        torch::Tensor gathered_out = gathered_out_buffer[i];

        run_fetch_kernel(
            static_x, static_stacked_grids, routing_map, port_map, spatial_map_buffer,
            config, cell_indices, gathered_out
        );

        py::dict tick_cache = bwd_cache[py::cast(i)].cast<py::dict>();
        torch::Tensor safe_layer_indices = tick_cache["safe_layer_indices"].cast<torch::Tensor>();
        torch::Tensor valid_cells = tick_cache["valid_cells"].cast<torch::Tensor>();
        torch::Tensor valid_idx = tick_cache["valid_idx"].cast<torch::Tensor>();

        py::dict tick_cell_params;
        for (std::pair<py::handle, py::handle> item : stacked_params) {
            torch::Tensor param_tensor = item.second.cast<torch::Tensor>();
            tick_cell_params[item.first] = param_tensor.index({safe_layer_indices});
        }

        std::vector<torch::Tensor> stacked_ingredients = torch::unbind(gathered_out, 0);
        py::tuple py_ingredients = py::cast(stacked_ingredients);
        py::tuple out_tuple = batched_forward(tick_cell_params, *py_ingredients);

        for (int p = 0; p < num_ports; ++p) {
            torch::Tensor port_output = out_tuple[p].cast<torch::Tensor>();
            static_stacked_grids[p].index_put_({valid_cells}, port_output.index({valid_idx}));
        }
    }
}

// BACKWARD PASS
void execute_backward(
    int num_ticks, int num_ports, int num_deps, int num_cells,
    py::function batched_backward_step,
    torch::Tensor active_cells_buffer, torch::Tensor gathered_out_buffer,
    torch::Tensor static_grad_x, std::vector<torch::Tensor> static_current_grad_grids,
    std::vector<torch::Tensor> static_grad_outputs,
    py::dict static_cell_grad_accumulators,
    py::dict bwd_cache, py::dict stacked_params, py::list dependencies
) {
    // 1. Clean the slate
    static_grad_x.zero_();
    for (auto& g : static_current_grad_grids) g.zero_();
    for (std::pair<py::handle, py::handle> item : static_cell_grad_accumulators) {
        item.second.cast<torch::Tensor>().zero_();
    }

    // 2. Move starting errors into the active grid
    for (int i = 0; i < num_ports; ++i) {
        static_current_grad_grids[i].slice(0, 0, num_cells).copy_(static_grad_outputs[i].slice(0, 0, num_cells));
    }

    // 3. The Reverse Tick Loop
    for (int t = num_ticks - 1; t >= 0; --t) {
        torch::Tensor cell_indices = active_cells_buffer[t];
        torch::Tensor gathered_tensors = gathered_out_buffer[t];

        py::dict tick_cache = bwd_cache[py::cast(t)].cast<py::dict>();
        torch::Tensor safe_layer_indices = tick_cache["safe_layer_indices"].cast<torch::Tensor>();
        torch::Tensor valid_idx = tick_cache["valid_idx"].cast<torch::Tensor>();
        torch::Tensor valid_cells = tick_cache["valid_cells"].cast<torch::Tensor>();

        py::dict cell_params;
        for (std::pair<py::handle, py::handle> item : stacked_params) {
            cell_params[item.first] = item.second.cast<torch::Tensor>().index({safe_layer_indices});
        }

        std::vector<torch::Tensor> tracked_stacked = torch::unbind(gathered_tensors, 0);
        py::tuple py_tracked_stacked = py::cast(tracked_stacked);

        std::vector<torch::Tensor> batched_grads_vec;
        for (int p = 0; p < num_ports; ++p) {
            batched_grads_vec.push_back(static_current_grad_grids[p].index({cell_indices}));
        }
        py::tuple py_batched_grads = py::cast(batched_grads_vec);

        // 4. Call PyTorch VJP Helper Natively
        py::tuple grad_returns = batched_backward_step(cell_params, py_tracked_stacked, py_batched_grads);

        // 5. Unpack and route the weight gradients safely
        py::dict grad_params = grad_returns[0].cast<py::dict>();
        for (std::pair<py::handle, py::handle> item : grad_params) {
            if (!item.second.is_none()) {
                torch::Tensor g = item.second.cast<torch::Tensor>();
                py::handle name = item.first;
                torch::Tensor acc = static_cell_grad_accumulators[name].cast<torch::Tensor>();
                torch::Tensor safe_g = g.index({valid_idx}).to(acc.scalar_type());
                acc.index_put_({valid_cells}, safe_g);
            }
        }

        // 6. Unpack and route the dependency gradients
        py::list deps_cache = tick_cache["deps"].cast<py::list>();
        for (int i = 0; i < num_deps; ++i) {
            py::tuple dep_tuple = dependencies[i].cast<py::tuple>();
            int source_port = dep_tuple[1].cast<int>();

            py::dict dep_cache = deps_cache[i].cast<py::dict>();
            // Note: grad_returns[0] is params, so grad_returns[i + 1] are the inputs
            torch::Tensor dep_grads = grad_returns[i + 1].cast<torch::Tensor>();
            torch::Tensor valid_grads = dep_grads.index({valid_idx});

            torch::Tensor x_targets = dep_cache["x_targets"].cast<torch::Tensor>();
            if (x_targets.numel() > 0) {
                torch::Tensor x_idx = dep_cache["x_idx"].cast<torch::Tensor>();
                static_grad_x.index_add_(1, x_targets, valid_grads.index({x_idx}).transpose(0, 1));
            }

            torch::Tensor grid_targets = dep_cache["grid_targets"].cast<torch::Tensor>();
            if (grid_targets.numel() > 0) {
                torch::Tensor grid_idx = dep_cache["grid_idx"].cast<torch::Tensor>();
                static_current_grad_grids[source_port].index_add_(0, grid_targets, valid_grads.index({grid_idx}));
            }
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("execute_forward", &execute_forward, "C++ Forward Pass Loop");
    m.def("execute_backward", &execute_backward, "C++ Backward Pass Loop");
}