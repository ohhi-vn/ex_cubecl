defmodule ExCubecl.MixProject do
  use Mix.Project

  def project do
    [
      app: :ex_cubecl,
      version: "0.1.0",
      elixir: "~> 1.19",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      rustler_crates: rustler_crates(),
      description: "ExCubeCL is a wrapper for CubeCL as backend for Nx",
      source_url: "https://github.com/manhvu/manhvu"
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:nx, "~> 0.12.0"},
      {:rustler, "~> 0.37"}
    ]
  end

  defp rustler_crates do
    [
      ex_cubecl_nif: [
        path: "native/ex_cubecl_nif",
        mode: if(Mix.env() == :prod, do: :release, else: :debug)
      ]
    ]
  end
end
