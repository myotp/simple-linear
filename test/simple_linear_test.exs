defmodule SimpleLinearTest do
  use ExUnit.Case
  doctest SimpleLinear

  test "greets the world" do
    assert SimpleLinear.hello() == :world
  end
end
