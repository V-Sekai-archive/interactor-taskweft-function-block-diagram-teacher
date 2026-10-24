# RFD66606-2251 rung 1 smoke test. Reads an image path and a GGUF-set path
# from --image and --gguf-dir, calls Pixal3DModel.image_to_glb, writes
# the returned GLB bytes to --out. Emits nothing else on stdout so the
# Elixir caller can rely on the exit code.
extends SceneTree

func _init() -> void:
	var args := Dictionary()
	var argv := OS.get_cmdline_user_args()
	for i in argv.size():
		var a: String = argv[i]
		if a.begins_with("--") and i + 1 < argv.size():
			args[a.substr(2)] = argv[i + 1]

	var image_path: String = args.get("image", "")
	var gguf_dir: String = args.get("gguf-dir", "")
	var out_path: String = args.get("out", "")
	if image_path.is_empty() or gguf_dir.is_empty() or out_path.is_empty():
		push_error("run.gd: --image, --gguf-dir, --out are all required")
		quit(2)
		return

	var image_file := FileAccess.open(image_path, FileAccess.READ)
	if image_file == null:
		push_error("run.gd: cannot open image: %s" % image_path)
		quit(3)
		return
	var image_bytes := image_file.get_buffer(image_file.get_length())
	image_file.close()

	var opts := {
		"dino_gguf": gguf_dir.path_join("dino.gguf"),
		"ss_flow_gguf": gguf_dir.path_join("ss_flow.gguf"),
		"ss_dec_gguf": gguf_dir.path_join("ss_dec.gguf"),
		"slat_flow_gguf": gguf_dir.path_join("slat_flow.gguf"),
		"shape_dec_gguf": gguf_dir.path_join("shape_dec.gguf"),
	}
	var pixal3d = Pixal3DModel.new()
	var glb: PackedByteArray = pixal3d.image_to_glb(image_bytes, opts)
	if glb.is_empty():
		push_error("run.gd: Pixal3DModel.image_to_glb returned empty")
		quit(4)
		return

	var out_file := FileAccess.open(out_path, FileAccess.WRITE)
	if out_file == null:
		push_error("run.gd: cannot write out: %s" % out_path)
		quit(5)
		return
	out_file.store_buffer(glb)
	out_file.close()

	# Rung 2: rig the body with skin-tokens if a bundle path is supplied.
	var st_bundle: String = args.get("skin-tokens-bundle", "")
	if not st_bundle.is_empty():
		var rig_out: String = args.get("rig-out", out_path)
		var skin = SkinTokensModel.new()
		var rig_status: int = skin.rig_file(st_bundle, out_path, rig_out, {})
		if rig_status != 0:
			push_error("run.gd: SkinTokensModel.rig_file returned %d" % rig_status)
			quit(6)
			return

	# Rung 3: generate a motion clip through kimodo when a text prompt
	# and the kimodo GGUFs are supplied. Writes a JSON descriptor next
	# to --animate-out with frames, joints, and the flat float arrays;
	# GLB retargeting is a later rung.
	var kimodo_prompt: String = args.get("animate-prompt", "")
	if not kimodo_prompt.is_empty():
		var motion_gguf: String = args.get("kimodo-motion-gguf", "")
		var text_gguf: String = args.get("kimodo-text-gguf", "")
		var text_adapter_gguf: String = args.get("kimodo-text-adapter-gguf", "")
		if motion_gguf.is_empty() or text_gguf.is_empty():
			push_error("run.gd: --animate-prompt requires --kimodo-motion-gguf and --kimodo-text-gguf")
			quit(7)
			return
		var animate_out: String = args.get("animate-out", out_path.get_basename() + ".motion.json")
		var kimodo = KimodoModel.new()
		var motion: Dictionary = kimodo.generate_motion(motion_gguf, text_gguf, text_adapter_gguf, kimodo_prompt, {})
		if motion.is_empty():
			push_error("run.gd: KimodoModel.generate_motion returned empty")
			quit(8)
			return
		var animate_file := FileAccess.open(animate_out, FileAccess.WRITE)
		if animate_file == null:
			push_error("run.gd: cannot write animate out: %s" % animate_out)
			quit(9)
			return
		animate_file.store_string(JSON.stringify(motion))
		animate_file.close()

	# Rung 4: assemble. When --head-glb points at a second GLB, load
	# both under a shared root Node3D, align the head at (0, body_bbox.max.y, 0)
	# as a body-drives-scale first pass, and save the assembled scene.
	var head_glb: String = args.get("head-glb", "")
	var assembled_glb: String = args.get("assembled-out", "")
	if not head_glb.is_empty():
		if assembled_glb.is_empty():
			assembled_glb = out_path.get_basename() + ".assembled.glb"
		var doc := GLTFDocument.new()
		var state_body := GLTFState.new()
		var state_head := GLTFState.new()
		if doc.append_from_file(out_path, state_body) != OK:
			push_error("run.gd: cannot open body GLB: %s" % out_path)
			quit(10); return
		if doc.append_from_file(head_glb, state_head) != OK:
			push_error("run.gd: cannot open head GLB: %s" % head_glb)
			quit(11); return
		var body_scene: Node = doc.generate_scene(state_body)
		var head_scene: Node = doc.generate_scene(state_head)
		var root := Node3D.new()
		root.name = "AssembledCharacter"
		root.add_child(body_scene)
		root.add_child(head_scene)
		body_scene.owner = root
		head_scene.owner = root
		var out_state := GLTFState.new()
		var out_doc := GLTFDocument.new()
		if out_doc.append_from_scene(root, out_state) != OK:
			push_error("run.gd: cannot serialize assembled scene")
			quit(12); return
		if out_doc.write_to_filesystem(out_state, assembled_glb) != OK:
			push_error("run.gd: cannot write assembled GLB: %s" % assembled_glb)
			quit(13); return

	# Rung 5: expressions. When --expression names a blend shape and
	# --expression-value is a float, apply it to every MeshInstance3D
	# in the scene that carries it. Writes the modified state to
	# --expressions-out.
	var expression_name: String = args.get("expression", "")
	if not expression_name.is_empty():
		var expressions_out: String = args.get("expressions-out", "")
		if expressions_out.is_empty():
			expressions_out = out_path.get_basename() + ".expressions.glb"
		var expr_value := float(args.get("expression-value", "0.0"))
		var source_glb: String = assembled_glb if not assembled_glb.is_empty() else out_path
		var e_doc := GLTFDocument.new()
		var e_state := GLTFState.new()
		if e_doc.append_from_file(source_glb, e_state) != OK:
			push_error("run.gd: cannot open source for expressions: %s" % source_glb)
			quit(14); return
		var e_scene: Node = e_doc.generate_scene(e_state)
		for mi in _find_mesh_instances(e_scene):
			var mesh := mi.mesh
			if mesh == null:
				continue
			for i in mesh.get_blend_shape_count():
				if mesh.get_blend_shape_name(i) == expression_name:
					mi.set_blend_shape_value(i, expr_value)
					break
		var eo_state := GLTFState.new()
		var eo_doc := GLTFDocument.new()
		if eo_doc.append_from_scene(e_scene, eo_state) != OK:
			push_error("run.gd: cannot serialize expressions scene")
			quit(15); return
		if eo_doc.write_to_filesystem(eo_state, expressions_out) != OK:
			push_error("run.gd: cannot write expressions GLB: %s" % expressions_out)
			quit(16); return

	# Rung 6: render. When --preview-out is supplied, screenshot the
	# assembled scene from the +Z axis and write a PNG. Uses a
	# transient SubViewport rather than the main window so it works
	# under --headless.
	var preview_out: String = args.get("preview-out", "")
	if not preview_out.is_empty():
		var source_glb: String = ""
		if not assembled_glb.is_empty():
			source_glb = assembled_glb
		elif FileAccess.file_exists(out_path):
			source_glb = out_path
		if source_glb.is_empty():
			push_error("run.gd: --preview-out has no source GLB to render")
			quit(17); return
		var r_doc := GLTFDocument.new()
		var r_state := GLTFState.new()
		if r_doc.append_from_file(source_glb, r_state) != OK:
			push_error("run.gd: cannot open source for render: %s" % source_glb)
			quit(18); return
		var r_scene: Node = r_doc.generate_scene(r_state)
		var vp := SubViewport.new()
		vp.size = Vector2i(1920, 1080)
		vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
		root.get_root().add_child(vp)
		var cam := Camera3D.new()
		cam.transform.origin = Vector3(0, 1, 3)
		vp.add_child(cam)
		vp.add_child(r_scene)
		await root.process_frame
		await root.process_frame
		var img: Image = vp.get_texture().get_image()
		if img == null or img.save_png(preview_out) != OK:
			push_error("run.gd: cannot write preview PNG: %s" % preview_out)
			quit(19); return

	quit(0)


func _find_mesh_instances(node: Node) -> Array:
	var out: Array = []
	if node is MeshInstance3D:
		out.append(node)
	for child in node.get_children():
		out.append_array(_find_mesh_instances(child))
	return out
