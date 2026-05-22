defmodule ExCubecl.Application do
  @moduledoc """
  Application callback for ExCubecl.

  Starts the GPU runtime supervision tree. The NIF resource management
  handles buffer cleanup automatically via Rustler's resource mechanism.
  """

  use Application

  @impl true
  def start(_type, _args) do
    children = [
      # GPU runtime state will be managed here in future phases.
      # For now, the NIF manages its own global state.
    ]

    Supervisor.start_link(children, strategy: :one_for_one, name: __MODULE__)
  end
end
