--------------------------------------------------------------------------------
-- GPU Profiler
--
-- Companion to "Widget Profiler" (dbg_widget_profiler.lua). That tool measures
-- CPU time spent *issuing* GL calls per widget; this one measures GPU-side cost,
-- which is decoupled from CPU time because the GPU runs asynchronously.
--
-- Pure-Lua, no engine patch required. Two profiling modes plus an always-on
-- VRAM readout:
--
--   * "widget" mode  -- per-widget CPU/GPU ms for each draw call-in, via
--                       gl.Finish() brackets. Finds which WIDGET is heavy.
--                       Inserts a glFinish around every hooked draw call-in,
--                       which serializes CPU<->GPU: absolute frame times are
--                       INFLATED while it runs. It is a diagnostic mode, not
--                       always-on telemetry. Relative attribution between
--                       widgets stays meaningful, which is what finds the hog.
--
--   * "overall" mode -- ONE glFinish at the very end of the frame measures the
--                       whole-frame GPU "tail": engine (CWorldDrawer) + all
--                       widgets. ~0 when CPU-bound, ~= total GPU frame time when
--                       GPU-bound. Use this first to decide CPU-bound vs
--                       GPU-bound, and whether the cost is the engine or a
--                       widget. Cheap: one finish/frame, no per-widget serialize.
--
--   * VRAM used/total via Spring.GetVidMemUsage() (driver-global, no per-resource
--     attribution; unavailable on Intel).
--
--   * VRAM pressure / paging: on NVIDIA we read the NVX eviction counters (via
--     the raw gl.GetNumber passthrough). When the driver is evicting resources
--     from VRAM the line turns red -- that means bad GPU perf is caused by VRAM
--     being full (thrashing over PCIe), which the used/total number alone cannot
--     tell you. AMD/Mesa lack the counter, so it falls back to a "near full"
--     heuristic; Intel has neither.
--------------------------------------------------------------------------------

local widget = widget ---@type Widget

function widget:GetInfo()
	return {
		name    = "GPU Profiler",
		desc    = "Per-widget GPU time (widget mode), whole-frame GPU time (overall mode) and live VRAM. Console: /gpuprofiler",
		author  = "bruno-dasilva",
		date    = "2026",
		license = "GNU GPL, v2 or later",
		layer   = -1000000, -- low layer => drawn last in reverse dispatch => overlay on top, and runs after all other draws
		handler = true,
		enabled = false,
	}
end

--------------------------------------------------------------------------------
-- Localised API
--------------------------------------------------------------------------------
local spEcho        = Spring.Echo
local spGetTimer    = Spring.GetTimer
local spDiffTimers  = Spring.DiffTimers
local spGetFPS      = Spring.GetFPS
local spGetVidMem   = Spring.GetVidMemUsage
local spGetMiniMapGeometry = Spring.GetMiniMapGeometry

local glFinish      = gl.Finish
local glText         = gl.Text
local glColor        = gl.Color
local glRect         = gl.Rect
local glBeginText    = gl.BeginText
local glEndText      = gl.EndText
local glGetViewSizes = gl.GetViewSizes

local mathMax    = math.max
local mathMin    = math.min
local mathFloor  = math.floor
local stringFormat = string.format
local stringGmatch = string.gmatch
local tableSort  = table.sort
local pairs, type, tonumber = pairs, type, tonumber

--------------------------------------------------------------------------------
-- Config / state
--------------------------------------------------------------------------------
local highres = false
if Spring.GetTimerMicros and Spring.GetConfigInt("UseHighResTimer", 0) == 1 then
	spGetTimer = Spring.GetTimerMicros
	highres = true
end

-- profiling modes
local MODE_OFF, MODE_WIDGET, MODE_OVERALL = 0, 1, 2
local profMode = MODE_OVERALL
local modeName = {
	[MODE_OFF]     = "off",
	[MODE_WIDGET]  = "widget (per-widget glFinish)",
	[MODE_OVERALL] = "overall (whole-frame GPU)",
}

local tick = 0.5 -- seconds between display refreshes
local maxRows = 24

-- Draw call-ins worth measuring on the GPU. We deliberately do NOT hook the
-- hundreds of non-draw call-ins the widget profiler hooks.
local DRAW_CALLINS = {
	"DrawGenesis",
	"DrawWorldPreParticles",
	"DrawWorldPreUnit",
	"DrawWorld",
	"DrawWorldRefraction",
	"DrawWorldReflection",
	"DrawWorldShadow",
	"DrawGroundPreForward",
	"DrawGroundPostDeferred",
	"DrawGroundDeferred",
	"DrawUnitsPostDeferred",
	"DrawFeaturesPostDeferred",
	"DrawShadowUnitsLua",
	"DrawShadowFeaturesLua",
	"DrawScreenEffects",
	"DrawScreenPost",
	"DrawScreenPre",
	"DrawScreen",
	"DrawInMiniMap",
	"DrawInMiniMapBackground",
}
local isDrawCallin = {}
for i = 1, #DRAW_CALLINS do isDrawCallin[DRAW_CALLINS[i]] = true end

local inHook = false
local hookFuncs = setmetatable({}, { __mode = "k" }) -- identity set of our hook closures

-- records[wname][callin] = rec ; recordList is a flat array for iteration
local records = {}
local recordList = {}
local hooked = false
local oldUpdateWidgetCallIn
local oldInsertWidget

-- display
local startTimer
local frameCount = 0
local displayList = {} -- sorted { name, ms, cpu, heaviest }
local totalMs = 0
local totalCpu = 0
local frameTailSum = 0 -- accumulated whole-frame GPU tail (seconds) this window
local frameGpuMs = 0   -- displayed avg whole-frame GPU tail (ms/frame)
local drawStartClock     -- timer at the frame's first draw call-in (DrawGenesis)
local drawStartValid = false
local drawCpuSum = 0   -- accumulated draw-frame CPU (seconds) this window
local cpuFrames = 0    -- draw frames sampled this window
local frameCpuMs = 0   -- displayed avg draw-frame CPU (ms): DrawGenesis..DrawScreen, excludes sim
local frameWallMs = 0  -- displayed avg draw-frame total (ms) = draw CPU + GPU tail

-- VRAM
local vramSupported = (spGetVidMem ~= nil)
local vramUsed, vramTotal = 0, 0

-- VRAM pressure / paging detection.
-- NVIDIA exposes eviction counters via GL_NVX_gpu_memory_info. The engine's
-- GetVidMemUsage ignores them, but we can read any GL enum through the raw
-- glGetFloatv passthrough gl.GetNumber. The counters climb only when the driver
-- evicts resources from VRAM to make room => paging under pressure. A rising
-- count during gameplay is a definitive "VRAM is full and thrashing" signal,
-- which the used/total number alone cannot tell you.
local glGetNumber = gl.GetNumber
local GL_GPU_MEMORY_INFO_EVICTION_COUNT_NVX = 0x904A
local GL_GPU_MEMORY_INFO_EVICTED_MEMORY_NVX = 0x904B
local evictSupported = (glGetNumber ~= nil) and (Platform ~= nil and Platform.gpuVendor == "Nvidia")
local prevEvictCount, prevEvictKB
local evictCountDelta = 0
local evictRateMBs = 0
local paging = false
local windowDt = tick

local title_colour  = "\255\160\255\160"
local totals_colour = "\255\200\200\255"

--------------------------------------------------------------------------------
-- Records
--------------------------------------------------------------------------------
local function GetRecord(wname, callin)
	local w = records[wname]
	if not w then
		w = {}
		records[wname] = w
	end
	local rec = w[callin]
	if not rec then
		rec = {
			wname = wname, callin = callin,
			t = 0, cpu = 0, n = 0, -- accumulated GPU / CPU seconds, frame-hits this window
		}
		w[callin] = rec
		recordList[#recordList + 1] = rec
	end
	return rec
end

--------------------------------------------------------------------------------
-- The hook (widget mode only)
--------------------------------------------------------------------------------
local function Hook(w, callin)
	local wname = w.whInfo.name
	if wname == "GPU Profiler" then
		return w[callin] -- never profile ourselves (our glFinish would nest)
	end

	local realFunc = w[callin]
	w["_gpuold_" .. callin] = realFunc
	local rec = GetRecord(wname, callin)

	local function timeDone(...)
		-- realFunc has just returned: CPU is done issuing GL commands for this pass
		local t1 = spGetTimer()
		rec.cpu = rec.cpu + spDiffTimers(t1, rec._t0, nil, highres)
		glFinish() -- block until this pass has actually finished on the GPU
		rec.t = rec.t + spDiffTimers(spGetTimer(), t1, nil, highres)
		rec.n = rec.n + 1
		inHook = false
		return ...
	end

	local function hook(...)
		-- only the per-widget "widget" mode brackets draws; off/overall pass through
		if inHook or profMode ~= MODE_WIDGET then
			return realFunc(...)
		end
		inHook = true
		glFinish() -- drain everything submitted before this widget
		rec._t0 = spGetTimer()
		return timeDone(realFunc(...))
	end

	hookFuncs[hook] = true
	return hook
end

--------------------------------------------------------------------------------
-- Install / remove hooks
--------------------------------------------------------------------------------
local function HookWidget(w)
	for i = 1, #DRAW_CALLINS do
		local callin = DRAW_CALLINS[i]
		if type(w[callin]) == "function" then
			w[callin] = Hook(w, callin)
		end
	end
end

local function StartHook()
	if hooked then return end
	local wh = widgetHandler

	for i = 1, #DRAW_CALLINS do
		local callin = DRAW_CALLINS[i]
		local list = wh[callin .. "List"]
		if list then
			for j = 1, #list do
				HookWidget(list[j])
			end
		end
	end

	-- catch call-ins (un)registered at runtime
	oldUpdateWidgetCallIn = wh.UpdateWidgetCallInRaw
	wh.UpdateWidgetCallInRaw = function(self, name, w)
		oldUpdateWidgetCallIn(self, name, w)
		if isDrawCallin[name] and type(w[name]) == "function" and not hookFuncs[w[name]] then
			w[name] = Hook(w, name)
			self:UpdateCallIn(name)
		end
	end

	-- catch widgets inserted later
	oldInsertWidget = wh.InsertWidgetRaw
	wh.InsertWidgetRaw = function(self, w)
		if w == nil then return end
		oldInsertWidget(self, w)
		HookWidget(w)
	end

	hooked = true
	spEcho("[GPU Profiler] hooked " .. #recordList .. " draw call-ins")
end

local function StopHook()
	if not hooked then return end
	local wh = widgetHandler

	for i = 1, #DRAW_CALLINS do
		local callin = DRAW_CALLINS[i]
		local list = wh[callin .. "List"]
		if list then
			for j = 1, #list do
				local w = list[j]
				if w["_gpuold_" .. callin] then
					w[callin] = w["_gpuold_" .. callin]
					w["_gpuold_" .. callin] = nil
				end
			end
		end
	end

	if oldUpdateWidgetCallIn then wh.UpdateWidgetCallInRaw = oldUpdateWidgetCallIn end
	if oldInsertWidget then wh.InsertWidgetRaw = oldInsertWidget end

	hooked = false
	spEcho("[GPU Profiler] unhooked")
end

--------------------------------------------------------------------------------
-- Lifecycle
--------------------------------------------------------------------------------
function widget:Initialize()
	-- defer hooking by one Update so every other widget has registered
end

function widget:Update()
	widgetHandler:RemoveWidgetCallIn("Update", self)
	StartHook()
	startTimer = spGetTimer()
end

function widget:Shutdown()
	StopHook()
end

--------------------------------------------------------------------------------
-- Console command:  /gpuprofiler [widget|overall|off|tick <n>]
--------------------------------------------------------------------------------
function widget:TextCommand(s)
	local tok = {}
	for w in stringGmatch(s, "%S+") do tok[#tok + 1] = w end
	if tok[1] ~= "gpuprofiler" then return end

	local arg = tok[2]
	if arg == "off" then
		profMode = MODE_OFF
	elseif arg == "widget" then
		profMode = MODE_WIDGET
	elseif arg == "overall" then
		profMode = MODE_OVERALL
	elseif arg == "tick" then
		tick = tonumber(tok[3]) or tick
		spEcho("[GPU Profiler] tick = " .. tick .. "s")
	else
		-- no/unknown arg: cycle off -> overall -> widget -> off
		if profMode == MODE_OFF then
			profMode = MODE_OVERALL
		elseif profMode == MODE_OVERALL then
			profMode = MODE_WIDGET
		else
			profMode = MODE_OFF
		end
	end
	spEcho("[GPU Profiler] mode: " .. modeName[profMode])
end

--------------------------------------------------------------------------------
-- Aggregate one window into the sorted display list
--------------------------------------------------------------------------------
local function Flush()
	local frames = mathMax(1, frameCount)
	local agg = {} -- wname -> { ms, cpu, heaviest, heaviestMs }

	for i = 1, #recordList do
		local rec = recordList[i]
		local a = agg[rec.wname]
		if not a then
			a = { ms = 0, cpu = 0, heaviest = "-", heaviestMs = 0 }
			agg[rec.wname] = a
		end

		local ms = (rec.t / frames) * 1000.0 -- seconds -> ms per frame
		a.ms = a.ms + ms
		a.cpu = a.cpu + (rec.cpu / frames) * 1000.0
		if ms > a.heaviestMs then
			a.heaviestMs = ms
			a.heaviest = rec.callin
		end

		rec.t, rec.cpu, rec.n = 0, 0, 0
	end

	displayList = {}
	totalMs, totalCpu = 0, 0
	for wname, a in pairs(agg) do
		displayList[#displayList + 1] = {
			name = wname, ms = a.ms, cpu = a.cpu, heaviest = a.heaviest,
		}
		totalMs = totalMs + a.ms
		totalCpu = totalCpu + a.cpu
	end
	tableSort(displayList, function(p, q) return p.ms > q.ms end)

	if vramSupported then
		local u, t = spGetVidMem()
		-- sanity-check: driver returns garbage on Intel / unsupported paths
		if t and t > 0 and t < 1e7 and u and u >= 0 and u <= t * 2 then
			vramUsed, vramTotal = u, t
		else
			vramSupported = false
		end
	end

	if evictSupported then
		local count = glGetNumber(GL_GPU_MEMORY_INFO_EVICTION_COUNT_NVX)
		local kb = glGetNumber(GL_GPU_MEMORY_INFO_EVICTED_MEMORY_NVX)
		if count and prevEvictCount then
			evictCountDelta = count - prevEvictCount
			local dMB = ((kb or prevEvictKB or 0) - (prevEvictKB or 0)) / 1024.0
			evictRateMBs = dMB / mathMax(windowDt, 0.001)
			paging = evictCountDelta > 0 -- any new evictions this window = paging
		end
		prevEvictCount, prevEvictKB = count, kb
	end

	frameGpuMs = (frameTailSum / frames) * 1000.0
	frameTailSum = 0
	frameCpuMs = (cpuFrames > 0) and ((drawCpuSum / cpuFrames) * 1000.0) or 0
	frameWallMs = frameCpuMs + frameGpuMs
	drawCpuSum, cpuFrames = 0, 0
	frameCount = 0
end

--------------------------------------------------------------------------------
-- Draw overlay
--------------------------------------------------------------------------------
local function txt(str, x, y, size)
	glText(str, x, y, size, "no")
end

-- Marks the start of the draw frame. DrawGenesis fires after the sim Update()
-- and before the world is rendered, so timing from here to DrawScreen captures
-- draw-frame CPU only -- no sim/GameFrame, no vsync idle.
function widget:DrawGenesis()
	if profMode == MODE_OVERALL then
		drawStartClock = spGetTimer()
		drawStartValid = true
	end
end

function widget:DrawScreen()
	frameCount = frameCount + 1

	-- "overall" mode: whole-frame GPU tail. We are layer -1000000 => drawn last,
	-- so by now the engine's CWorldDrawer AND every widget have submitted all
	-- their GL work. This glFinish blocks until the GPU drains it: ~0 when
	-- CPU-bound, ~= total GPU frame time when GPU-bound. One finish/frame.
	if profMode == MODE_OVERALL then
		-- tA = all CPU submission for this draw frame is done (our DrawScreen is last)
		local tA = spGetTimer()
		if drawStartValid then
			-- Frame CPU = work submitting GL calls: DrawGenesis (draw-frame start) .. now
			drawCpuSum = drawCpuSum + spDiffTimers(tA, drawStartClock, nil, highres)
			cpuFrames = cpuFrames + 1
			drawStartValid = false
		end
		-- GPU tail = GPU work still outstanding once the CPU is done submitting
		glFinish()
		frameTailSum = frameTailSum + spDiffTimers(spGetTimer(), tA, nil, highres)
	end

	local dt = spDiffTimers(spGetTimer(), startTimer, nil, highres)
	if dt >= tick then
		startTimer = spGetTimer()
		windowDt = dt
		Flush()
	end

	local vsx, vsy = glGetViewSizes()
	local fontSize = mathMax(11, mathFloor(vsy / 95))
	local line = fontSize + 4
	-- anchor x to the right of the minimap; keep y where it was
	local mmx, _, mmw, _, mmMin, mmMax = spGetMiniMapGeometry()
	local x
	if mmx and not mmMin and not mmMax then
		x = mathFloor(mmx + mmw + fontSize)
	else
		x = mathFloor(vsx * 0.012) -- fallback when minimap is hidden/maximized
	end
	local y = mathFloor(vsy * 0.9)
	local colCpu = x + fontSize * 13
	local colGpu = x + fontSize * 19
	local colTot = x + fontSize * 25
	local colHit = x + fontSize * 31

	glColor(1, 1, 1, 1)
	glBeginText()

	-- ---- header ----
	txt(title_colour .. "GPU Profiler  \255\200\200\200(" .. (Platform and Platform.gpuVendor or "?") .. ")", x, y, fontSize)
	y = y - line

	-- VRAM bar
	if vramSupported and vramTotal > 0 then
		local frac = mathMin(1, vramUsed / vramTotal)
		local barW = fontSize * 22
		local barH = fontSize
		glColor(0.15, 0.15, 0.15, 0.8)
		glRect(x, y - 2, x + barW, y - 2 + barH)
		-- green -> red as it fills
		glColor(0.3 + 0.7 * frac, 0.9 - 0.7 * frac, 0.15, 0.9)
		glRect(x, y - 2, x + barW * frac, y - 2 + barH)
		glColor(1, 1, 1, 1)
		txt(stringFormat("VRAM %d / %d MB  (%.0f%%)", vramUsed, vramTotal, frac * 100),
			x + barW + fontSize, y, fontSize)
	else
		txt(totals_colour .. "VRAM: n/a (Intel/unsupported)", x, y, fontSize)
	end
	y = y - line

	-- VRAM pressure / paging line
	if evictSupported then
		if paging then
			glColor(1, 0.35, 0.25, 1)
			txt(stringFormat("VRAM PRESSURE: paging  (%d evictions, %.0f MB/s) -- bad perf is from full VRAM", evictCountDelta, evictRateMBs), x, y, fontSize)
		else
			glColor(0.5, 0.85, 0.5, 1)
			txt("VRAM pressure: none (driver not evicting)", x, y, fontSize)
		end
		glColor(1, 1, 1, 1)
		y = y - line
	elseif vramSupported and vramTotal > 0 and (vramUsed / vramTotal) > 0.95 then
		glColor(1, 0.7, 0.2, 1)
		txt("VRAM near full -- pressure likely (no eviction counter on this GPU)", x, y, fontSize)
		glColor(1, 1, 1, 1)
		y = y - line
	end

	txt(totals_colour .. stringFormat("FPS %d   mode: ", spGetFPS()) .. modeName[profMode]
		.. "   \255\160\160\160/gpuprofiler [overall|widget|off]", x, y, fontSize)
	y = y - line
	y = y - mathFloor(line * 0.3)

	-- ---- body, per mode ----
	if profMode == MODE_OFF then
		txt(totals_colour .. "profiling paused", x, y, fontSize)

	elseif profMode == MODE_OVERALL then
		-- whole-frame CPU vs GPU (engine + widgets)
		local heat = mathMin(1, frameGpuMs / 8.0)
		txt(totals_colour .. stringFormat("Frame CPU: %.2f ms", frameCpuMs), x, y, fontSize)
		glColor(1, 1 - 0.7 * heat, 1 - 0.7 * heat, 1)
		txt(stringFormat("Frame GPU tail: %.2f ms", frameGpuMs), x + fontSize * 13, y, fontSize)
		glColor(1, 1, 1, 1)
		txt(title_colour .. stringFormat("total %.2f ms", frameWallMs), x + fontSize * 28, y, fontSize)
		y = y - line
		txt(totals_colour .. "  GPU ~0    => CPU-bound: GPU keeps up, look at CPU/sim (use Widget Profiler)", x, y, fontSize)
		y = y - line
		txt(totals_colour .. "  GPU high  => switch to 'widget' mode; if widgets are cheap it's the engine", x, y, fontSize)
		y = y - line
		txt(totals_colour .. "              (CWorldDrawer: terrain/units/particles/water) -> RenderDoc/Tracy", x, y, fontSize)
		y = y - line
		txt("\255\160\160\160  Frame CPU = work submitting GL calls (DrawGenesis..DrawScreen).", x, y, fontSize)
		y = y - line
		txt("\255\160\160\160  Frame GPU tail = GPU work left after submission. Both exclude sim & vsync idle.", x, y, fontSize)

	else -- MODE_WIDGET
		txt("\255\255\200\120glFinish active: frame times are INFLATED; compare widgets relative to each other", x, y, fontSize)
		y = y - line
		y = y - mathFloor(line * 0.3)

		-- table header
		txt(title_colour .. "widget", x, y, fontSize)
		txt(title_colour .. "CPU ms/f", colCpu, y, fontSize)
		txt(title_colour .. "GPU ms/f", colGpu, y, fontSize)
		txt(title_colour .. "total", colTot, y, fontSize)
		txt(title_colour .. "heaviest", colHit, y, fontSize)
		y = y - line

		-- rows
		local rows = mathMin(maxRows, #displayList)
		for i = 1, rows do
			local d = displayList[i]
			if d.ms < 0.005 then break end
			-- redden the expensive ones (by GPU)
			local heat = mathMin(1, d.ms / 2.0)
			glColor(1, 1 - 0.7 * heat, 1 - 0.7 * heat, 1)
			txt(d.name, x, y, fontSize)
			txt(stringFormat("%.3f", d.ms), colGpu, y, fontSize)
			glColor(0.6, 0.82, 1.0, 1) -- cpu in cyan to distinguish
			txt(stringFormat("%.3f", d.cpu), colCpu, y, fontSize)
			glColor(1, 1, 1, 1) -- total = cpu + gpu = (now - t0)
			txt(stringFormat("%.3f", d.cpu + d.ms), colTot, y, fontSize)
			glColor(0.6, 0.6, 0.6, 1)
			txt(d.heaviest, colHit, y, fontSize)
			y = y - line
		end

		-- totals
		y = y - mathFloor(line * 0.3)
		glColor(1, 1, 1, 1)
		txt(totals_colour .. stringFormat("total  CPU %.3f  /  GPU %.3f  /  sum %.3f  ms/frame (widget draws; serialized)", totalCpu, totalMs, totalCpu + totalMs), x, y, fontSize)
	end

	glColor(1, 1, 1, 1)
	glEndText()
end
