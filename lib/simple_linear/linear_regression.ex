defmodule SimpleLinear.LinearRegression do
  import Nx.Defn

  def run() do
    Application.put_env(:nx, :default_backend, EXLA.Backend)
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
    |> Nx.reshape({})
    |> Nx.to_number()
  end

  def train(epochs, data) do
    init_params = init_random_params()

    for _ <- 1..epochs, reduce: init_params do
      acc ->
        data
        |> Enum.take(200)
        |> Enum.reduce(
          acc,
          fn batch, cur_params ->
            {inp, tar} = Enum.unzip(batch)
            x = Nx.tensor(inp)
            y = Nx.tensor(tar)
            update(cur_params, x, y)
          end
        )
    end
  end

  # y = mx + b
  defn init_random_params do
    key = Nx.Random.key(57)
    {m, new_key} = Nx.Random.normal(key, 0.0, 0.1, shape: {1, 1})
    {b, _new_key} = Nx.Random.normal(new_key, 0.0, 0.1, shape: {1})
    {m, b}
  end

  defn update({m, b} = params, inp, tar) do
    {grad_m, grad_b} = grad(params, &loss(&1, inp, tar))

    {
      m - grad_m * 0.01,
      b - grad_b * 0.01
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
