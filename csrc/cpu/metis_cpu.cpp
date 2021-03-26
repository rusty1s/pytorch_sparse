#include "metis_cpu.h"

#ifdef WITH_METIS
#include <metis.h>
#endif

#ifdef WITH_MTMETIS
#include <mtmetis.h>
#endif

#include "utils.h"

torch::Tensor partition_cpu(torch::Tensor rowptr, torch::Tensor col,
                            torch::optional<torch::Tensor> optional_value,
                            torch::optional<torch::Tensor> optional_node_weight,
                            int64_t num_parts, bool recursive) {
#ifdef WITH_METIS
  CHECK_CPU(rowptr);
  CHECK_CPU(col);

  if (optional_value.has_value()) {
    CHECK_CPU(optional_value.value());
    CHECK_INPUT(optional_value.value().dim() == 1);
    CHECK_INPUT(optional_value.value().numel() == col.numel());
  }

  if (optional_node_weight.has_value()) {
    CHECK_CPU(optional_node_weight.value());
    CHECK_INPUT(optional_node_weight.value().dim() == 1);
    CHECK_INPUT(optional_node_weight.value().numel() == rowptr.numel() - 1);
  }

  int64_t nvtxs = rowptr.numel() - 1;
  int64_t ncon = 1;
  auto *xadj = rowptr.data_ptr<int64_t>();
  auto *adjncy = col.data_ptr<int64_t>();

  int64_t *adjwgt = NULL;
  if (optional_value.has_value())
    adjwgt = optional_value.value().data_ptr<int64_t>();

  int64_t *vwgt = NULL;
  if (optional_node_weight.has_value())
    vwgt = optional_node_weight.value().data_ptr<int64_t>();

  int64_t objval = -1;
  auto part = torch::empty(nvtxs, rowptr.options());
  auto part_data = part.data_ptr<int64_t>();

  if (recursive) {
    METIS_PartGraphRecursive(&nvtxs, &ncon, xadj, adjncy, vwgt, NULL, adjwgt,
                             &num_parts, NULL, NULL, NULL, &objval, part_data);
  } else {
    METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, vwgt, NULL, adjwgt,
                        &num_parts, NULL, NULL, NULL, &objval, part_data);
  }

  return part;
#else
  AT_ERROR("Not compiled with METIS support");
#endif
}

// needs mt-metis installed via:
// ./configure --shared --edges64bit --vertices64bit --weights64bit
//             --partitions64bit
torch::Tensor
mt_partition_cpu(torch::Tensor rowptr, torch::Tensor col,
                 torch::optional<torch::Tensor> optional_value,
                 torch::optional<torch::Tensor> optional_node_weight,
                 int64_t num_parts, bool recursive, int64_t num_workers) {
#ifdef WITH_MTMETIS
  CHECK_CPU(rowptr);
  CHECK_CPU(col);
  if (optional_value.has_value()) {
    CHECK_CPU(optional_value.value());
    CHECK_INPUT(optional_value.value().dim() == 1);
    CHECK_INPUT(optional_value.value().numel() == col.numel());
  }

  if (optional_node_weight.has_value()) {
    CHECK_CPU(optional_node_weight.value());
    CHECK_INPUT(optional_node_weight.value().dim() == 1);
    CHECK_INPUT(optional_node_weight.value().numel() == rowptr.numel() - 1);
  }

  mtmetis_vtx_type nvtxs = rowptr.numel() - 1;
  mtmetis_vtx_type ncon = 1;
  mtmetis_adj_type *xadj = (mtmetis_adj_type *)rowptr.data_ptr<int64_t>();
  mtmetis_vtx_type *adjncy = (mtmetis_vtx_type *)col.data_ptr<int64_t>();
  mtmetis_wgt_type *adjwgt = NULL;

  if (optional_value.has_value())
    adjwgt = optional_value.value().data_ptr<int64_t>();

  mtmetis_wgt_type *vwgt = NULL;
  if (optional_node_weight.has_value())
    vwgt = optional_node_weight.value().data_ptr<int64_t>();

  mtmetis_pid_type nparts = num_parts;
  mtmetis_wgt_type objval = -1;
  auto part = torch::empty(nvtxs, rowptr.options());
  mtmetis_pid_type *part_data = (mtmetis_pid_type *)part.data_ptr<int64_t>();

  double *opts = mtmetis_init_options();
  opts[MTMETIS_OPTION_NTHREADS] = num_workers;

  if (recursive) {
    MTMETIS_PartGraphRecursive(&nvtxs, &ncon, xadj, adjncy, vwgt, NULL, adjwgt,
                               &nparts, NULL, NULL, opts, &objval, part_data);
  } else {
    MTMETIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, vwgt, NULL, adjwgt,
                          &nparts, NULL, NULL, opts, &objval, part_data);
  }

  return part;
#else
  AT_ERROR("Not compiled with MTMETIS support");
#endif
}
