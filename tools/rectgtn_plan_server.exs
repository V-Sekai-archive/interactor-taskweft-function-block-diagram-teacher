# SPDX-License-Identifier: MIT
# Copyright (c) 2026 K. S. Ernest (iFire) Lee
#
# One JSON object per line: {"domain": <domain JSON-LD>} -> {"ok": true, "plan": <plan JSON>}
IO.puts(Jason.encode!(%{ready: true}))

for line <- IO.stream(:stdio, :line) do
  case Jason.decode(line) do
    {:ok, %{"domain" => domain}} ->
      case Taskweft.plan(Jason.encode!(domain)) do
        {:ok, plan} -> IO.puts(Jason.encode!(%{ok: true, plan: plan}))
        {:error, reason} -> IO.puts(Jason.encode!(%{ok: false, error: inspect(reason)}))
      end

    _ ->
      IO.puts(Jason.encode!(%{ok: false, error: "a line must be {\"domain\": ...}"}))
  end
end
