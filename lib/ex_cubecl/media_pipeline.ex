defmodule ExCubecl.MediaPipeline do
  @moduledoc """
  A GenServer behaviour for real-time media processing pipelines.

  Designed for livestreaming, camera effects, and other real-time use cases
  where frames are processed as they arrive.

  ## Example

      defmodule MyLivestream do
        use ExCubecl.MediaPipeline

        def handle_frame(frame, state) do
          frame
          |> ExCubecl.Filter.apply(:denoise, strength: 0.5)
          |> ExCubecl.Video.overlay(state.logo, x: 10, y: 10)
          |> ExCubecl.Transcode.write_frame(state.encoder)
          {:ok, state}
        end
      end

  ## Callbacks

    * `handle_frame/2` — called for each incoming frame. Receives the frame
      and current state, must return `{:ok, new_state}` or `{:error, reason}`.
  """

  @callback handle_frame(ExCubecl.VideoFrame.t(), state :: term()) ::
              {:ok, new_state :: term()} | {:error, term()}

  @doc false
  defmacro __using__(_opts) do
    quote do
      use GenServer
      @behaviour ExCubecl.MediaPipeline

      @impl GenServer
      def init(state) do
        {:ok, state}
      end

      @impl GenServer
      def handle_info({:frame, frame}, state) do
        case handle_frame(frame, state) do
          {:ok, new_state} ->
            {:noreply, new_state}

          {:error, reason} ->
            {:stop, reason, state}
        end
      end

      defoverridable init: 1, handle_info: 2
    end
  end

  @doc """
  Starts a media pipeline GenServer.

  ## Options

    * `:name` — register the process under a name
    * `:source` — media source path or reference
    * `:encoder` — encoder reference from `Transcode.start/2`
  """
  @spec start_link(module(), term(), keyword()) :: GenServer.on_start()
  def start_link(module, init_arg, opts \\ []) do
    GenServer.start_link(module, init_arg, opts)
  end

  @doc "Pushes a frame into the pipeline for processing."
  @spec push_frame(GenServer.server(), ExCubecl.VideoFrame.t()) :: :ok
  def push_frame(server, %ExCubecl.VideoFrame{} = frame) do
    send(server, {:frame, frame})
    :ok
  end
end
