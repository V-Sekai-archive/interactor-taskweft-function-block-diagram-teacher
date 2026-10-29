defmodule TaskweftFbdTeacher.Runner.Character do
  @moduledoc """
  RFD66606-2251 rung 1 runner. Reads an image plus a GGUF-set directory, boots
  the assembled Godot binary named by `GODOT_ASSEMBLY_BIN`, runs the
  character workflow GDScript in headless mode, and returns the GLB bytes.

  A missing `GODOT_ASSEMBLY_BIN` is a refusal, not a silent skip: the
  runner never falls back to a stub. Same shape as the compiler runner
  in this repository.
  """

  @behaviour TaskweftFbdTeacher.Runner

  defstruct [:bin, :script, :sha, timeout: 600_000]

  @impl true
  def start(opts) do
    bin = Keyword.get_lazy(opts, :bin, fn -> System.get_env("GODOT_ASSEMBLY_BIN") end)

    cond do
      bin in [nil, ""] ->
        {:error, {:no_godot_assembly_bin, "set GODOT_ASSEMBLY_BIN or pass :bin"}}

      not File.regular?(bin) ->
        {:error, {:no_godot_assembly_bin, bin}}

      true ->
        script = Keyword.get_lazy(opts, :script, fn -> default_script() end)

        cond do
          not File.regular?(script) ->
            {:error, {:no_script, script}}

          true ->
            {:ok,
             %__MODULE__{
               bin: Path.expand(bin),
               script: Path.expand(script),
               sha: sha(bin),
               timeout: Keyword.get(opts, :timeout, 600_000)
             }}
        end
    end
  end

  @impl true
  def provenance(%__MODULE__{} = c),
    do: %{godot_assembly_bin: c.bin, godot_assembly_sha: c.sha, script: c.script}

  @impl true
  def stop(%__MODULE__{}), do: :ok

  @impl true
  def run(%__MODULE__{} = c, %{image: image, gguf_dir: gguf_dir, out: out} = job) do
    cond do
      not File.regular?(image) ->
        {:refused, {:no_image, image}}

      not File.dir?(gguf_dir) ->
        {:refused, {:no_gguf_dir, gguf_dir}}

      Map.has_key?(job, :skin_tokens_bundle) and not File.regular?(job.skin_tokens_bundle) ->
        {:refused, {:no_skin_tokens_bundle, job.skin_tokens_bundle}}

      Map.has_key?(job, :animate_prompt) and
          (not is_binary(Map.get(job, :kimodo_motion_gguf)) or
             not File.regular?(job.kimodo_motion_gguf)) ->
        {:refused, {:no_kimodo_motion_gguf, Map.get(job, :kimodo_motion_gguf)}}

      Map.has_key?(job, :animate_prompt) and
          (not is_binary(Map.get(job, :kimodo_text_gguf)) or
             not File.regular?(job.kimodo_text_gguf)) ->
        {:refused, {:no_kimodo_text_gguf, Map.get(job, :kimodo_text_gguf)}}

      Map.has_key?(job, :head_glb) and not File.regular?(job.head_glb) ->
        {:refused, {:no_head_glb, job.head_glb}}

      true ->
        base = [
          "--headless",
          "--script",
          c.script,
          "++",
          "--image",
          Path.expand(image),
          "--gguf-dir",
          Path.expand(gguf_dir),
          "--out",
          Path.expand(out)
        ]

        rig_args =
          case job do
            %{skin_tokens_bundle: bundle} ->
              rig_out = Map.get(job, :rig_out, out)

              [
                "--skin-tokens-bundle",
                Path.expand(bundle),
                "--rig-out",
                Path.expand(rig_out)
              ]

            _ ->
              []
          end

        animate_args =
          case job do
            %{animate_prompt: prompt} ->
              motion = Path.expand(job.kimodo_motion_gguf)
              text = Path.expand(job.kimodo_text_gguf)
              adapter = job |> Map.get(:kimodo_text_adapter_gguf, "") |> to_string()
              animate_out = Map.get(job, :animate_out, out <> ".motion.json")

              [
                "--animate-prompt",
                prompt,
                "--kimodo-motion-gguf",
                motion,
                "--kimodo-text-gguf",
                text,
                "--kimodo-text-adapter-gguf",
                if(adapter == "", do: "", else: Path.expand(adapter)),
                "--animate-out",
                Path.expand(animate_out)
              ]

            _ ->
              []
          end

        assemble_args =
          case job do
            %{head_glb: head} ->
              assembled_out = Map.get(job, :assembled_out, out <> ".assembled.glb")

              [
                "--head-glb",
                Path.expand(head),
                "--assembled-out",
                Path.expand(assembled_out)
              ]

            _ ->
              []
          end

        expressions_args =
          case job do
            %{expression: name} ->
              value = Map.get(job, :expression_value, 1.0)
              expressions_out = Map.get(job, :expressions_out, out <> ".expressions.glb")

              [
                "--expression",
                to_string(name),
                "--expression-value",
                to_string(value),
                "--expressions-out",
                Path.expand(expressions_out)
              ]

            _ ->
              []
          end

        render_args =
          case job do
            %{preview_out: preview} -> ["--preview-out", Path.expand(preview)]
            _ -> []
          end

        args = base ++ rig_args ++ animate_args ++ assemble_args ++ expressions_args ++ render_args

        case System.cmd(c.bin, args, stderr_to_stdout: true) do
          {out_str, 0} ->
            {:ok,
             %{
               glb_path: Path.expand(out),
               rig_path:
                 case job do
                   %{skin_tokens_bundle: _} -> Path.expand(Map.get(job, :rig_out, out))
                   _ -> nil
                 end,
               animate_path:
                 case job do
                   %{animate_prompt: _} ->
                     Path.expand(Map.get(job, :animate_out, out <> ".motion.json"))

                   _ ->
                     nil
                 end,
               assembled_path:
                 case job do
                   %{head_glb: _} ->
                     Path.expand(Map.get(job, :assembled_out, out <> ".assembled.glb"))

                   _ ->
                     nil
                 end,
               expressions_path:
                 case job do
                   %{expression: _} ->
                     Path.expand(Map.get(job, :expressions_out, out <> ".expressions.glb"))

                   _ ->
                     nil
                 end,
               preview_path: Map.get(job, :preview_out) && Path.expand(job.preview_out),
               godot_log: out_str
             }}

          {out_str, code} ->
            {:refused, {:godot_exit, code, out_str}}
        end
    end
  end

  defp default_script do
    Application.app_dir(:taskweft_fbd_teacher, ["priv", "character", "run.gd"])
  end

  defp sha(bin) do
    case System.cmd("shasum", ["-a", "256", bin], stderr_to_stdout: true) do
      {out, 0} -> out |> String.split() |> List.first()
      _ -> nil
    end
  end
end
