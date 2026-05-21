defmodule ExCubecl.MixProject do
  use Mix.Project

  @version "0.2.2"
  @source_url "https://github.com/ohhi-vn/ex_cubecl"

  def project do
    [
      app: :ex_cubecl,
      version: @version,
      elixir: "~> 1.19",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      rustler_crates: rustler_crates(),
      description: description(),
      package: package(),
      docs: docs(),
      name: "ExCubecl",
      source_url: @source_url,
      homepage_url: @source_url,
      elixirc_paths: elixirc_paths(Mix.env())
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:rustler, "~> 0.37", runtime: false},
      {:ex_doc, "~> 0.40", only: :dev, runtime: false}
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

  defp description do
    """
    ExCubecl is a GPU compute runtime for Elixir powered by CubeCL via Rust NIFs.
    Provides GPU buffer management, kernel execution, async command submission,
    and pipeline orchestration. Designed for AI inference, media processing,
    and realtime GPU effects on mobile and desktop.
    """
  end

  defp package do
    [
      name: "ex_cubecl",
      files: ~w(
        lib
        priv
        native/ex_cubecl_nif/Cargo.toml
        native/ex_cubecl_nif/Cargo.lock
        native/ex_cubecl_nif/src
        native/ex_cubecl_nif/include
        guides
        examples
        .formatter.exs
        mix.exs
        mix.lock
        README.md
        LICENSE
      ),
      licenses: ["Apache-2.0"],
      links: %{
        "GitHub" => @source_url,
        "Changelog" => "#{@source_url}/blob/main/CHANGELOG.md",
        "Sponsor" => @source_url
      },
      maintainers: ["Manh Vu"]
    ]
  end

  defp docs do
    [
      main: "readme",
      source_url: @source_url,
      source_ref: "v#{@version}",
      extras: [
        "README.md",
        "guides/01_getting_started.md",
        "guides/02_buffers.md",
        "guides/03_kernels.md",
        "guides/04_async_pipelines.md",
        "guides/05_mobile.md",
        "guides/06_examples.md"
      ],
      groups_for_extras: [
        Guides: ~r/guides\/[0-9]+_.*\.md/,
        Examples: ~r/examples\/.*\.exs/
      ],
      groups_for_modules: [
        "GPU Runtime": [ExCubecl],
        "NIF Stubs": [ExCubecl.NIF]
      ],
      skip_undefined_reference_warnings_on: ["guides/05_mobile.md"],
      before_closing_body_tag: &before_closing_body_tag/1
    ]
  end

  # Add a small footer to the docs
  defp before_closing_body_tag(:html) do
    """
    <script>
      // Add version badge to docs
      var footer = document.querySelector('.footer');
      if (footer) {
        footer.innerHTML += '<br><small>ExCubecl v#{@version}</small>';
      }
    </script>
    """
  end

  defp before_closing_body_tag(_), do: ""
end
