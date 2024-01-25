defmodule SimpleLinear.LinearRegression do
  import Nx.Defn

  def run() do
    Nx.default_backend(EXLA.Backend)

    target_m = :rand.normal(0.0, 10.0)
    target_b = :rand.normal(0.0, 5.0)
    target_fn = fn x -> target_m * x + target_b end

    data =
      Stream.repeatedly(fn -> for _ <- 1..32, do: :rand.uniform() * 10 end)
      |> Stream.map(fn x -> Enum.zip(x, Enum.map(x, target_fn)) end)

    IO.puts("Target m: #{target_m}\tTarget b: #{target_b}\n")
    {m, b} = train(100, data)
    IO.puts("Learned m: #{to_scalar(m)}\tLearned b: #{to_scalar(b)}")
  end

  # Nx.to_scalar/1
  def to_scalar(t) do
    t
    |> Nx.squeeze()
    |> Nx.to_number()
  end

  # 训练主函数
  def train(epochs, data) do
    init_params = init_random_params()

    # 循环epochs次
    Enum.reduce(1..epochs, init_params, fn _, acc ->
      data
      # 每次循环当中来200组数据, 再次循环
      |> Enum.take(200)
      |> Enum.reduce(
        acc,
        fn batch, cur_params ->
          # 每一组数据都是32个具体的数值
          {input, tar} = Enum.unzip(batch)
          x = Nx.tensor(input)
          y = Nx.tensor(tar)
          update(cur_params, x, y, 0.001)
        end
      )
    end)
  end

  # 开局随机选择m,b就可以了
  # y = mx + b
  defn init_random_params do
    key = Nx.Random.key(57)
    {m, new_key} = Nx.Random.normal(key, 0.0, 0.1, shape: {1, 1})
    {b, _new_key} = Nx.Random.normal(new_key, 0.0, 0.1, shape: {1})
    {m, b}
  end

  defn update({m, b} = params, inp, tar, learning_rate) do
    {grad_m, grad_b} = Nx.Defn.grad(params, &loss(&1, inp, tar))

    {
      m - grad_m * learning_rate,
      b - grad_b * learning_rate
    }
  end

  # mean-squared error (MSE)
  defn loss(params, x, y) do
    y_pred = predict(params, x)
    Nx.mean(Nx.pow(y - y_pred, 2))
  end

  defn predict({m, b}, x) do
    m * x + b
  end
end
