--------------------------------------------------------------------------------
-- GPU Profiler
--
-- Companion to "Widget Profiler" (dbg_widget_profiler.lua). That tool measures
-- CPU time spent *issuing* GL calls per widget; this one measures GPU-side cost,
-- which is decoupled from CPU time because the GPU runs asynchronously.
--
-- Three pure-Lua signals (no engine patch required):
--   1. VRAM used/total           -- Spring.GetVidMemUsage()  (always on)
--   2. GPU ms per widget draw    -- gl.Finish() brackets      (mode: time)
--   3. Overdraw (samples) /widget-- occlusion queries         (mode: overdraw)
--
-- Caveats (by design, see README block at bottom):
--   * "time" mode inserts glFinish around every hooked draw call-in, which
--     serializes CPU<->GPU. Absolute frame times are INFLATED while it runs;
--     it is a diagnostic mode, not always-on telemetry. Relative attribution
--     between widgets stays meaningful, which is what finds the hog.
--   * VRAM is the driver's global view (no per-resource attribution) and is
--     unavailable on Intel.
--------------------------------------------------------------------------------

local widget = widget ---@type Widget

function widget:GetInfo()
	return {
		name    = "GPU Profiler",
		desc    = "Per-widget GPU time (glFinish), overdraw (occlusion queries) and live VRAM usage. Console: /gpuprofiler",
		author  = "BAR",
		date    = "2026",
		license = "GNU GPL, v2 or later",
		layer   = -1000000, -- low layer => drawn last in reverse dispatch => overlay on top
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

local glFinish      = gl.Finish
local glCreateQuery = gl.CreateQuery
local glDeleteQuery = gl.DeleteQuery
local glRunQuery    = gl.RunQuery
local glGetQuery    = gl.GetQuery

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
local pairs, next, type, tonumber, pcall = pairs, next, type, tonumber, pcall

--------------------------------------------------------------------------------
-- Config / state
--------------------------------------------------------------------------------
local highres = false
if Spring.GetTimerMicros and Spring.GetConfigInt("UseHighResTimer", 0) == 1 then
	spGetTimer = Spring.GetTimerMicros
	highres = true
end

local overdrawSupported = (glCreateQuery ~= nil and glRunQuery ~= nil and glGetQuery ~= nil)

-- profiling modes
local MODE_OFF, MODE_TIME, MODE_OVERDRAW = 0, 1, 2
local profMode = MODE_TIME -- starts profiling time as soon as the widget is enabled
local modeName = { [MODE_OFF] = "off", [MODE_TIME] = "time (glFinish)", [MODE_OVERDRAW] = "overdraw (occlusion)" }

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
local displayList = {} -- sorted { name, ms, samples, heaviest }
local totalMs = 0
local totalSamples = 0

-- VRAM
local vramSupported = (spGetVidMem ~= nil)
local vramUsed, vramTotal = 0, 0

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
			t = 0, n = 0,            -- accumulated GPU seconds / frame-hits this window
			samples = 0, sN = 0,     -- accumulated occlusion samples / hits this window
			queries = nil,           -- { q0, q1 } lazily created
			pending = { false, false },
			slot = 1,
		}
		w[callin] = rec
		recordList[#recordList + 1] = rec
	end
	return rec
end

--------------------------------------------------------------------------------
-- The hook
--------------------------------------------------------------------------------
local function Hook(w, callin)
	local wname = w.whInfo.name
	if wname == "GPU Profiler" then
		return w[callin] -- never profile ourselves (our glFinish/queries would nest)
	end

	local realFunc = w[callin]
	w["_gpuold_" .. callin] = realFunc
	local rec = GetRecord(wname, callin)

	local function timeDone(...)
		glFinish() -- block until this pass has actually finished on the GPU
		rec.t = rec.t + spDiffTimers(spGetTimer(), rec._t0, nil, highres)
		rec.n = rec.n + 1
		inHook = false
		return ...
	end

	local function hook(...)
		if inHook or profMode == MODE_OFF then
			return realFunc(...)
		end
		inHook = true

		if profMode == MODE_TIME then
			glFinish() -- drain everything submitted before this widget
			rec._t0 = spGetTimer()
			return timeDone(realFunc(...))
		end

		-- MODE_OVERDRAW: count fragments via an occlusion query.
		-- Double-buffered: read the result from 2 frames ago (never stalls),
		-- because gl.GetQuery() blocks on GL_QUERY_RESULT.
		local q = rec.queries
		if not q then
			local q0, q1 = glCreateQuery(), glCreateQuery()
			if not q0 or not q1 then
				profMode = MODE_OFF -- driver refused queries; bail safely
				inHook = false
				return realFunc(...)
			end
			q = { q0, q1 }
			rec.queries = q
		end

		local slot = rec.slot
		if rec.pending[slot] then
			local samples = glGetQuery(q[slot])
			if samples then
				rec.samples = rec.samples + samples
				rec.sN = rec.sN + 1
			end
		end

		-- gl.RunQuery(q, fn, ...) runs fn(...) between glBeginQuery/glEndQuery.
		-- It discards fn's return values, which is fine for draw call-ins.
		local ok = pcall(glRunQuery, q[slot], realFunc, ...)
		if not ok then
			-- recursion/error inside the query (rare); fall back uncounted
			realFunc(...)
			rec.pending[slot] = false
		else
			rec.pending[slot] = true
		end
		rec.slot = (slot % 2) + 1

		inHook = false
		return
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

	-- free GPU query objects
	for i = 1, #recordList do
		local q = recordList[i].queries
		if q then
			if q[1] then glDeleteQuery(q[1]) end
			if q[2] then glDeleteQuery(q[2]) end
			recordList[i].queries = nil
		end
	end

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
-- Console command:  /gpuprofiler [time|overdraw|off|tick <n>]
--------------------------------------------------------------------------------
function widget:TextCommand(s)
	local tok = {}
	for w in stringGmatch(s, "%S+") do tok[#tok + 1] = w end
	if tok[1] ~= "gpuprofiler" then return end

	local arg = tok[2]
	if arg == "off" then
		profMode = MODE_OFF
	elseif arg == "time" then
		profMode = MODE_TIME
	elseif arg == "overdraw" then
		if overdrawSupported then
			profMode = MODE_OVERDRAW
		else
			spEcho("[GPU Profiler] overdraw unsupported (no occlusion query extension)")
		end
	elseif arg == "tick" then
		tick = tonumber(tok[3]) or tick
		spEcho("[GPU Profiler] tick = " .. tick .. "s")
	else
		-- no/unknown arg: cycle off -> time -> overdraw -> off
		if profMode == MODE_OFF then
			profMode = MODE_TIME
		elseif profMode == MODE_TIME then
			profMode = overdrawSupported and MODE_OVERDRAW or MODE_OFF
		else
			profMode = MODE_OFF
		end
	end
	spEcho("[GPU Profiler] mode: " .. modeName[profMode])
end

--------------------------------------------------------------------------------
-- Aggregate one window of samples into the sorted display list
--------------------------------------------------------------------------------
local function Flush()
	local frames = mathMax(1, frameCount)
	local agg = {} -- wname -> { ms, samples, heaviest, heaviestMs }

	for i = 1, #recordList do
		local rec = recordList[i]
		local a = agg[rec.wname]
		if not a then
			a = { ms = 0, samples = 0, heaviest = "-", heaviestMs = 0 }
			agg[rec.wname] = a
		end

		local ms = (rec.t / frames) * 1000.0 -- seconds -> ms per frame
		a.ms = a.ms + ms
		if ms > a.heaviestMs then
			a.heaviestMs = ms
			a.heaviest = rec.callin
		end
		-- occlusion samples averaged over the hits we actually read back
		if rec.sN > 0 then
			a.samples = a.samples + (rec.samples / rec.sN)
		end

		rec.t, rec.n, rec.samples, rec.sN = 0, 0, 0, 0
	end

	displayList = {}
	totalMs, totalSamples = 0, 0
	for wname, a in pairs(agg) do
		displayList[#displayList + 1] = {
			name = wname, ms = a.ms, samples = a.samples, heaviest = a.heaviest,
		}
		totalMs = totalMs + a.ms
		totalSamples = totalSamples + a.samples
	end

	if profMode == MODE_OVERDRAW then
		tableSort(displayList, function(p, q) return p.samples > q.samples end)
	else
		tableSort(displayList, function(p, q) return p.ms > q.ms end)
	end

	if vramSupported then
		local u, t = spGetVidMem()
		-- sanity-check: driver returns garbage on Intel / unsupported paths
		if t and t > 0 and t < 1e7 and u and u >= 0 and u <= t * 2 then
			vramUsed, vramTotal = u, t
		else
			vramSupported = false
		end
	end

	frameCount = 0
end

--------------------------------------------------------------------------------
-- Draw overlay
--------------------------------------------------------------------------------
local function txt(str, x, y, size)
	glText(str, x, y, size, "no")
end

function widget:DrawScreen()
	frameCount = frameCount + 1

	if spDiffTimers(spGetTimer(), startTimer, nil, highres) >= tick then
		startTimer = spGetTimer()
		Flush()
	end

	local vsx, vsy = glGetViewSizes()
	local fontSize = mathMax(11, mathFloor(vsy / 95))
	local line = fontSize + 4
	local x = mathFloor(vsx * 0.012)
	local y = mathFloor(vsy * 0.86)
	local colMs = x + fontSize * 16
	local colHit = x + fontSize * 23

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

	txt(totals_colour .. stringFormat("FPS %d   mode: ", spGetFPS()) .. modeName[profMode]
		.. "   \255\160\160\160/gpuprofiler [time|overdraw|off]", x, y, fontSize)
	y = y - line

	if profMode == MODE_TIME then
		txt("\255\255\200\120glFinish active: frame times are INFLATED; compare widgets relative to each other", x, y, fontSize)
		y = y - line
	end

	-- ---- table header ----
	y = y - mathFloor(line * 0.3)
	txt(title_colour .. "widget", x, y, fontSize)
	if profMode == MODE_OVERDRAW then
		txt(title_colour .. "Msamp/f", colMs, y, fontSize)
	else
		txt(title_colour .. "GPU ms/f", colMs, y, fontSize)
		txt(title_colour .. "heaviest", colHit, y, fontSize)
	end
	y = y - line

	-- ---- rows ----
	if profMode == MODE_OFF then
		txt(totals_colour .. "profiling paused", x, y, fontSize)
	else
		local rows = mathMin(maxRows, #displayList)
		for i = 1, rows do
			local d = displayList[i]
			if profMode == MODE_OVERDRAW then
				if d.samples < 1 then break end
				glColor(1, 1, 1, 1)
				txt(d.name, x, y, fontSize)
				txt(stringFormat("%.2f", d.samples / 1.0e6), colMs, y, fontSize)
			else
				if d.ms < 0.005 then break end
				-- redden the expensive ones
				local heat = mathMin(1, d.ms / 2.0)
				glColor(1, 1 - 0.7 * heat, 1 - 0.7 * heat, 1)
				txt(d.name, x, y, fontSize)
				txt(stringFormat("%.3f", d.ms), colMs, y, fontSize)
				glColor(0.6, 0.6, 0.6, 1)
				txt(d.heaviest, colHit, y, fontSize)
			end
			y = y - line
		end

		-- totals
		y = y - mathFloor(line * 0.3)
		glColor(1, 1, 1, 1)
		if profMode == MODE_OVERDRAW then
			txt(totals_colour .. stringFormat("total %.2f Msamples/frame", totalSamples / 1.0e6), x, y, fontSize)
		else
			txt(totals_colour .. stringFormat("total %.3f GPU ms/frame (widget draws; serialized)", totalMs), x, y, fontSize)
		end
	end

	glColor(1, 1, 1, 1)
	glEndText()
end
