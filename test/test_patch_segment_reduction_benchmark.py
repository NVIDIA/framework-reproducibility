class SegmentReductionOpBenchmark(test.Benchmark):

  outer_dim_options = [2**x for x in range(9, 14, 2)]
  ratio_options = [2**x for x in range(1, 6, 2)]
  inner_dim_options = [2**x for x in range(9, 14, 2)]
  # randomly generated sizes with less alignments
  inner_dim_options += [
      1120, 1215, 1856, 1302, 1329, 1531, 1313, 1672, 1851, 1584
  ]
  dtype_options = [np.float32, np.float64]
  options = (outer_dim_options, ratio_options, inner_dim_options, dtype_options)
  # pylint: disable=g-long-lambda
  op_functors = [lambda vc, vs, seg_ids:
                 ("sorted", math_ops.segment_sum(vc, vs)),
                 lambda vc, vs, seg_ids:
                 ("unsorted",
                  math_ops.unsorted_segment_sum(vc, vs, seg_ids[-1]+1))]
  # pylint: enable=g-long-lambda
  repeat = 10

  def _npTypeToStr(self, t):
    if t == np.float32:
      return "fp32"
    if t == np.float64:
      return "fp64"

  def _runGraph(self, op_functor, outer_dim, ratio, inner_dim, dtype):
    output_outer_dim = int(outer_dim / ratio)
    const = np.random.randint(5, size=(outer_dim, inner_dim))
    seg_ids = np.sort(np.random.randint(output_outer_dim, size=outer_dim))
    vs = variables.Variable(seg_ids.astype(np.int32))
    with ops.device("/gpu:0"):
      vc = variables.Variable(const.astype(dtype))
    name, op = op_functor(vc, vs, seg_ids)
    with session.Session() as sess:
      variables.global_variables_initializer().run()
      r = self.run_op_benchmark(
          sess,
          op,
          min_iters=self.repeat,
          name="_".join(
              map(str,
                  [name, outer_dim, ratio, inner_dim,
                   self._npTypeToStr(dtype)])))
    return name, r["wall_time"]

  def benchmarkSegmentSumGPU(self):
    if not test.is_gpu_available(cuda_only=True):
      return
    for outer_dim, ratio, inner_dim, dtype in itertools.product(*self.options):
      op_functor = self.op_functors[0]
      with ops.Graph().as_default():
        self._runGraph(op_functor, outer_dim, ratio, inner_dim, dtype)

  def benchmarkUnsortedSegmentSumGPU(self):
    if not test.is_gpu_available(cuda_only=True):
      return
    for outer_dim, ratio, inner_dim, dtype in itertools.product(*self.options):
      op_functor = self.op_functors[1]
      with ops.Graph().as_default():
        self._runGraph(op_functor, outer_dim, ratio, inner_dim, dtype)