import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import type { ChangeEventHandler, ClipboardEvent, KeyboardEvent, MouseEvent as ReactMouseEvent } from "react"
import { Moon, RefreshCw, Sun, Trophy, ZoomIn, ZoomOut } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Textarea } from "@/components/ui/textarea"
import { cn } from "@/lib/utils"
import { type LanguageKey, type Snippet, LEVEL_SEGMENTS, SNIPPETS } from "@/data/snippets"

import "./App.css"

type TypingSession = {
  id: string
  language: LanguageKey
  snippetId: string
  snippetTitle: string
  wpm: number
  accuracy: number
  errors: number
  characters: number
  durationSeconds: number
  timestamp: number
}

type SpeedPoint = {
  time: number
  wpm: number
}

const STORAGE_KEY = "coding-master-history"
const UNLOCK_STORAGE_KEY = "coding-master-level"

const LANGUAGE_OPTIONS: Array<{ value: LanguageKey; label: string }> = [
  { value: "javascript", label: "JavaScript" },
  { value: "python", label: "Python" },
  { value: "java", label: "Java" },
  { value: "cpp", label: "C++" },
  { value: "c", label: "C" },
]

const languageLabel = (value: LanguageKey) =>
  LANGUAGE_OPTIONS.find((option) => option.value === value)?.label ?? value.toUpperCase()

const THEME_STORAGE_KEY = "coding-master-theme"
const HITS_STORAGE_KEY = "coding-master-hits"
const CODE_SCALE_MIN = 0.85
const CODE_SCALE_MAX = 1.35
const CODE_SCALE_STEP = 0.05
const MIN_CODE_PANE_WIDTH = 480
const MAX_CODE_PANE_WIDTH = 1200

type TournamentTier = {
  label: string
  description: string
  start: number
  end: number
  requiredWpm: number
  index: number
}

const tierBlueprints: Array<{ label: string; description: string; requiredWpm: number }> = [
  {
    label: "Level 0 · Dojo Initiate",
    description: "Simple loops, array sums, and string reversals to build precise muscle memory.",
    requiredWpm: 0,
  },
  {
    label: "Level 1 · Algorithm Apprentice",
    description: "Core algorithms unlocked: binary search, BFS, DFS, and first dynamic programming drills.",
    requiredWpm: 28,
  },
  {
    label: "Level 2 · Data Structure Duelist",
    description: "Intermediate duels: priority queues, topological order, and disjoint set tournaments.",
    requiredWpm: 38,
  },
  {
    label: "Level 3 · Grandmaster Summit",
    description: "Championship code: segment trees, KMP, knapsack, and all-pairs shortest paths.",
    requiredWpm: 50,
  },
]

const TOURNAMENT_TIERS: TournamentTier[] = (() => {
  const tiers: TournamentTier[] = []
  let cursor = 1
  tierBlueprints.forEach((tier, index) => {
    const span = LEVEL_SEGMENTS[index] ?? 0
    const start = cursor
    const end = cursor + Math.max(span, 0) - 1
    tiers.push({ ...tier, start, end, index })
    cursor = end + 1
  })
  return tiers
})()

const getTierInfo = (position: number) => {
  const fallback = TOURNAMENT_TIERS[TOURNAMENT_TIERS.length - 1]
  for (let index = 0; index < TOURNAMENT_TIERS.length; index += 1) {
    const tier = TOURNAMENT_TIERS[index]
    if (position >= tier.start && position <= tier.end) {
      return tier
    }
  }
  return { ...fallback, index: TOURNAMENT_TIERS.length - 1 }
}

const selectSnippet = (
  language: LanguageKey,
  maxTierIndex: number,
  excludeId?: string,
): Snippet => {
  const clampedTier = Math.max(0, Math.min(maxTierIndex, TOURNAMENT_TIERS.length - 1))
  const allowedEnd = TOURNAMENT_TIERS[clampedTier].end
  const pool = SNIPPETS[language].filter((candidate) => {
    const parts = candidate.id.split("-")
    const numeric = Number.parseInt(parts[1] ?? "0", 10)
    if (Number.isNaN(numeric) || numeric <= 0) {
      return false
    }
    if (numeric > allowedEnd) {
      return false
    }
    if (excludeId && candidate.id === excludeId) {
      return false
    }
    return true
  })
  if (pool.length > 0) {
    return pool[Math.floor(Math.random() * pool.length)]
  }
  return SNIPPETS[language][0]
}

const loadHistory = (): TypingSession[] => {
  if (typeof window === "undefined") {
    return []
  }
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) {
      return []
    }
    const parsed = JSON.parse(raw) as TypingSession[]
    return Array.isArray(parsed) ? parsed : []
  } catch (error) {
    console.error("Failed to read typing history", error)
    return []
  }
}

const storeHistory = (entries: TypingSession[]) => {
  if (typeof window === "undefined") {
    return
  }
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(entries))
  } catch (error) {
    console.error("Failed to persist typing history", error)
  }
}

const evaluateInput = (snippet: string, input: string) => {
  let correct = 0
  for (let index = 0; index < input.length; index += 1) {
    if (snippet[index] === input[index]) {
      correct += 1
    }
  }
  const mismatches = Math.max(0, input.length - correct)
  const accuracy = input.length ? (correct / input.length) * 100 : 100
  return { correct, mismatches, accuracy }
}

const formatNumber = (value: number, digits = 1) =>
  Number.isFinite(value) ? value.toFixed(digits) : (0).toFixed(digits)

const formatDuration = (seconds: number) => {
  if (!Number.isFinite(seconds)) {
    return "0s"
  }
  const minutes = Math.floor(seconds / 60)
  const remainder = Math.round(seconds % 60)
  return minutes > 0 ? `${minutes}m ${remainder}s` : `${remainder}s`
}

const createImprovementTips = (
  payload: {
    wpm: number
    accuracy: number
    errors: number
    characters: number
    bestWpm?: number
  },
): string[] => {
  const { wpm, accuracy, errors, characters, bestWpm } = payload
  if (characters === 0) {
    return ["Start typing to unlock personalised coaching tips."]
  }
  const tips: string[] = []
  if (accuracy < 92) {
    tips.push("Slow down slightly and focus on matching punctuation accurately.")
  }
  if (errors > Math.max(3, characters * 0.08)) {
    tips.push("Re-read tricky lines aloud; it helps encode the indentation and braces.")
  }
  if (wpm < 35) {
    tips.push("Drill small snippets repeatedly to build rhythmic muscle memory.")
  } else if (bestWpm && wpm + 2 < bestWpm) {
    tips.push(`You are ${formatNumber(bestWpm - wpm, 1)} WPM away from your best run - aim for smoother bursts.`)
  } else if (!bestWpm || wpm >= bestWpm) {
    tips.push("Outstanding pace! You are matching or exceeding your personal best.")
  }
  if (!tips.length) {
    tips.push("Balanced session - maintain this consistency and gradually raise your goal speed.")
  }
  return tips.slice(0, 3)
}

const SpeedGraph = ({ data }: { data: SpeedPoint[] }) => {
  if (data.length < 2) {
    return (
      <div className="flex h-44 w-full items-center justify-center rounded-lg border border-dashed text-sm text-muted-foreground">
        Start typing to visualise your live speed profile.
      </div>
    )
  }

  const width = 540
  const height = 176
  const padding = 24
  const maxTime = data[data.length - 1]?.time || 1
  const maxWpm = Math.max(60, ...data.map((point) => point.wpm))
  const coordinates = data.map((point) => {
    const x =
      padding +
      (maxTime === 0 ? 0 : (point.time / maxTime) * (width - padding * 2))
    const y =
      height -
      padding -
      (maxWpm === 0 ? 0 : (point.wpm / maxWpm) * (height - padding * 2))
    return { x, y }
  })
  const points = coordinates.map((coord) => `${coord.x},${coord.y}`).join(" ")
  const lastX = coordinates[coordinates.length - 1]?.x ?? padding
  const areaPoints = `${padding},${height - padding} ${points} ${lastX},${height - padding}`
  const gridLines = [0.25, 0.5, 0.75]

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-44 w-full">
      <defs>
        <linearGradient id="speedFill" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0%" stopColor="hsl(var(--primary))" stopOpacity="0.4" />
          <stop offset="100%" stopColor="hsl(var(--primary))" stopOpacity="0" />
        </linearGradient>
      </defs>

      {gridLines.map((ratio) => {
        const y = height - padding - ratio * (height - padding * 2)
        return (
          <line
            key={ratio}
            x1={padding}
            y1={y}
            x2={width - padding}
            y2={y}
            stroke="hsl(var(--muted-foreground))"
            strokeOpacity="0.15"
            strokeWidth={1}
          />
        )
      })}

      <line
        x1={padding}
        y1={height - padding}
        x2={width - padding}
        y2={height - padding}
        stroke="hsl(var(--border))"
        strokeWidth={1.5}
        strokeLinecap="round"
      />

      <polygon points={areaPoints} fill="url(#speedFill)" />

      <polyline
        points={points}
        fill="none"
        stroke="hsl(var(--primary))"
        strokeWidth={3}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

const App = () => {
  const [activeTab, setActiveTab] = useState<"practice" | "history">("practice")
  const [language, setLanguage] = useState<LanguageKey>("javascript")
  const [unlockedTierIndex, setUnlockedTierIndex] = useState(0)
  const [snippet, setSnippet] = useState<Snippet>(() => selectSnippet("javascript", 0))
  const [userInput, setUserInput] = useState("")
  const [startTime, setStartTime] = useState<number | null>(null)
  const [elapsedMs, setElapsedMs] = useState(0)
  const [isRunning, setIsRunning] = useState(false)
  const [hasFinished, setHasFinished] = useState(false)
  const [speedSeries, setSpeedSeries] = useState<SpeedPoint[]>([])
  const [history, setHistory] = useState<TypingSession[]>(() => loadHistory())
  const [isDarkMode, setIsDarkMode] = useState(false)
  const [codeScale, setCodeScale] = useState(1)
  const [showResultsModal, setShowResultsModal] = useState(false)
  const [lastSession, setLastSession] = useState<TypingSession | null>(null)
  const [siteHits, setSiteHits] = useState(0)
  const [unlockNotice, setUnlockNotice] = useState<string | null>(null)
  const [codePaneWidth, setCodePaneWidth] = useState<number | null>(null)

  const codePaneRef = useRef<HTMLDivElement | null>(null)
  const caretRef = useRef<HTMLSpanElement | null>(null)
  const dragStateRef = useRef<{ startX: number; startWidth: number } | null>(null)
  const mouseMoveListenerRef = useRef<((event: MouseEvent) => void) | null>(null)
  const mouseUpListenerRef = useRef<(() => void) | null>(null)

  const isCodePaneExpanded = (codePaneWidth ?? 0) > 720
  const practiceLayoutClass = isCodePaneExpanded
    ? "flex flex-col gap-6"
    : "grid gap-6 lg:grid-cols-[minmax(0,1.6fr),minmax(0,1fr)]"

  const practiceLayoutStyle = useMemo(() => {
    if (isCodePaneExpanded || !codePaneWidth) {
      return undefined
    }
    const clamped = Math.min(codePaneWidth, MAX_CODE_PANE_WIDTH)
    return { gridTemplateColumns: `minmax(0, ${Math.round(clamped)}px) minmax(0, 1fr)` }
  }, [isCodePaneExpanded, codePaneWidth])

  const codePaneStyle = useMemo(() => {
    if (!codePaneWidth) {
      return undefined
    }
    if (isCodePaneExpanded) {
      return { width: "100%", maxWidth: "100%" }
    }
    const clamped = Math.min(codePaneWidth, MAX_CODE_PANE_WIDTH)
    return { width: `${Math.round(clamped)}px`, maxWidth: "100%" }
  }, [codePaneWidth, isCodePaneExpanded])

  const snippetLength = snippet.code.length

  const metrics = useMemo(() => evaluateInput(snippet.code, userInput), [snippet, userInput])

  const completion = snippetLength ? Math.min(1, userInput.length / snippetLength) : 0
  const elapsedMinutes = elapsedMs / 1000 / 60
  const wpm = elapsedMinutes > 0 ? (metrics.correct / 5) / elapsedMinutes : 0
  const accuracy = metrics.accuracy
  const errors = metrics.mismatches
  const remainingCharacters = Math.max(0, snippetLength - userInput.length)
  const elapsedSeconds = elapsedMs / 1000

  const totalTournamentSnippets = SNIPPETS[language].length
  const snippetPosition = useMemo(() => {
    const parts = snippet.id.split("-")
    const numeric = Number.parseInt(parts[1] ?? "1", 10)
    if (Number.isNaN(numeric)) {
      return 1
    }
    return Math.max(1, Math.min(numeric, totalTournamentSnippets))
  }, [snippet.id, totalTournamentSnippets])

  const tierDetails = useMemo(() => getTierInfo(snippetPosition), [snippetPosition])
  const nextTier = tierDetails.index < TOURNAMENT_TIERS.length - 1 ? TOURNAMENT_TIERS[tierDetails.index + 1] : undefined

  const bestForLanguage = useMemo(
    () =>
      history
        .filter((entry) => entry.language === language)
        .reduce<TypingSession | undefined>((best, entry) => {
          if (!best || entry.wpm > best.wpm) {
            return entry
          }
          return best
        }, undefined),
    [history, language],
  )

  const historySummary = useMemo(() => {
    if (!history.length) {
      return null
    }
    const totalSessions = history.length
    const totalWpm = history.reduce((total, entry) => total + entry.wpm, 0)
    const totalAccuracy = history.reduce((total, entry) => total + entry.accuracy, 0)
    const languagesPractised = new Set(history.map((entry) => entry.language)).size
    return {
      totalSessions,
      averageWpm: totalWpm / totalSessions,
      averageAccuracy: totalAccuracy / totalSessions,
      languagesPractised,
    }
  }, [history])

  const improvementTips = useMemo(
    () =>
      createImprovementTips({
        wpm,
        accuracy,
        errors,
        characters: userInput.length,
        bestWpm: bestForLanguage?.wpm,
      }),
    [wpm, accuracy, errors, userInput.length, bestForLanguage?.wpm],
  )

  const formattedSiteHits = useMemo(() => siteHits.toLocaleString(), [siteHits])

  useEffect(() => {
    if (typeof window === "undefined") {
      return
    }
    const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY)
    if (storedTheme === "dark" || storedTheme === "light") {
      setIsDarkMode(storedTheme === "dark")
    } else if (window.matchMedia("(prefers-color-scheme: dark)").matches) {
      setIsDarkMode(true)
    }
    const storedHits = Number.parseInt(window.localStorage.getItem(HITS_STORAGE_KEY) ?? "0", 10)
    if (!Number.isNaN(storedHits) && storedHits > 0) {
      setSiteHits(storedHits)
    }
    const storedTier = Number.parseInt(window.localStorage.getItem(UNLOCK_STORAGE_KEY) ?? "0", 10)
    if (!Number.isNaN(storedTier)) {
      setUnlockedTierIndex(Math.max(0, Math.min(storedTier, TOURNAMENT_TIERS.length - 1)))
    }
  }, [])

  useEffect(() => {
    if (typeof window === "undefined") {
      return
    }
    const root = window.document.documentElement
    if (isDarkMode) {
      root.classList.add("dark")
    } else {
      root.classList.remove("dark")
    }
    window.localStorage.setItem(THEME_STORAGE_KEY, isDarkMode ? "dark" : "light")
  }, [isDarkMode])

  useEffect(() => {
    return () => {
      if (mouseMoveListenerRef.current) {
        document.removeEventListener("mousemove", mouseMoveListenerRef.current)
      }
      if (mouseUpListenerRef.current) {
        document.removeEventListener("mouseup", mouseUpListenerRef.current)
      }
      document.body.style.removeProperty("user-select")
      dragStateRef.current = null
    }
  }, [])

  const updateUserInput = (rawValue: string) => {
    if (hasFinished) {
      return
    }
    const sanitized = rawValue.slice(0, snippetLength)
    if (!isRunning && sanitized.length > 0) {
      const now = Date.now()
      setStartTime(now)
      setIsRunning(true)
      setElapsedMs(0)
      setSpeedSeries([{ time: 0, wpm: 0 }])
    }
    setUserInput(sanitized)
  }

  const handleLanguageChange = (value: string) => {
    const nextLanguage = value as LanguageKey
    setLanguage(nextLanguage)
    loadNewSnippet(nextLanguage, false)
  }

  const loadNewSnippet = useCallback(
    (
      targetLanguage: LanguageKey,
      excludeCurrent = true,
      tierOverride?: number,
    ) => {
      const effectiveTier = Math.max(
        0,
        Math.min(typeof tierOverride === "number" ? tierOverride : unlockedTierIndex, unlockedTierIndex),
      )
      const excludeId = excludeCurrent && snippet.language === targetLanguage ? snippet.id : undefined
      const nextSnippet = selectSnippet(targetLanguage, effectiveTier, excludeId)
      setSnippet(nextSnippet)
      setUserInput("")
      setStartTime(null)
      setElapsedMs(0)
      setIsRunning(false)
      setHasFinished(false)
      setSpeedSeries([])
      setShowResultsModal(false)
      setLastSession(null)
      setUnlockNotice(null)
    },
    [snippet.id, snippet.language, unlockedTierIndex],
  )

  useEffect(() => {
    if (tierDetails.index > unlockedTierIndex) {
      loadNewSnippet(language, false, unlockedTierIndex)
    }
  }, [tierDetails.index, unlockedTierIndex, language, loadNewSnippet])

  const handleNewSnippet = () => {
    loadNewSnippet(language, true)
  }

  const handleNextRound = () => {
    handleNewSnippet()
  }

  const handleDismissResults = () => {
    setShowResultsModal(false)
    setUnlockNotice(null)
  }

  const handleToggleTheme = () => {
    setIsDarkMode((previous) => !previous)
  }

  const adjustCodeScale = (delta: number) => {
    setCodeScale((previous) => {
      const next = Math.min(CODE_SCALE_MAX, Math.max(CODE_SCALE_MIN, previous + delta))
      return Number(next.toFixed(2))
    })
  }

  const resetCodeScale = () => {
    setCodeScale(1)
  }

  const abortSession = () => {}
const handleResizeMouseDown = (event: ReactMouseEvent<HTMLDivElement>) => {
    if (!codePaneRef.current) {
      return
    }
    event.preventDefault()
    const pane = codePaneRef.current
    dragStateRef.current = {
      startX: event.clientX,
      startWidth: pane.offsetWidth,
    }
    const previousUserSelect = document.body.style.userSelect
    const handleMouseMove = (moveEvent: MouseEvent) => {
      if (!dragStateRef.current || !codePaneRef.current) {
        return
      }
      const parentWidth =
        codePaneRef.current.offsetParent instanceof HTMLElement
          ? codePaneRef.current.offsetParent.getBoundingClientRect().width
          : MAX_CODE_PANE_WIDTH
      const minWidth = Math.max(320, Math.min(MIN_CODE_PANE_WIDTH, parentWidth * 0.6))
      const maxWidth = Math.max(minWidth, Math.min(MAX_CODE_PANE_WIDTH, parentWidth))
      const delta = moveEvent.clientX - dragStateRef.current.startX
      const nextWidth = Math.max(minWidth, Math.min(dragStateRef.current.startWidth + delta, maxWidth))
      setCodePaneWidth(nextWidth)
    }
    const handleMouseUp = () => {
      dragStateRef.current = null
      document.removeEventListener("mousemove", handleMouseMove)
      document.removeEventListener("mouseup", handleMouseUp)
      document.body.style.userSelect = previousUserSelect
      mouseMoveListenerRef.current = null
      mouseUpListenerRef.current = null
    }
    mouseMoveListenerRef.current = handleMouseMove
    mouseUpListenerRef.current = handleMouseUp
    document.addEventListener("mousemove", handleMouseMove)
    document.addEventListener("mouseup", handleMouseUp)
    document.body.style.userSelect = "none"
  }

  const handleTyping: ChangeEventHandler<HTMLTextAreaElement> = (event) => {
    if (hasFinished) {
      return
    }
    updateUserInput(event.target.value)
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    const isClipboardCombo =
      (event.metaKey || event.ctrlKey) && ["c", "v", "x"].includes(event.key.toLowerCase())

    if (isClipboardCombo) {
      event.preventDefault()
      return
    }

    if (event.key === "Backspace" || event.key === "Delete") {
      event.preventDefault()
      return
    }

    if (event.key === "Tab") {
      event.preventDefault()
      if (hasFinished) {
        return
      }
      const cursor = userInput.length
      const remainingCapacity = snippetLength - cursor
      if (remainingCapacity <= 0) {
        return
      }
      const remaining = snippet.code.slice(cursor)
      const match = remaining.match(/^[ \t]+/)
      if (!match) {
        return
      }
      const addition = match[0].slice(0, remainingCapacity)
      if (!addition.length) {
        return
      }
      updateUserInput(userInput + addition)
    }
  }

  const blockClipboardEvent = (event: ClipboardEvent<HTMLTextAreaElement>) => {
    event.preventDefault()
  }

  const completeSession = useCallback(() => {
    if (hasFinished || snippetLength === 0) {
      return
    }
    const timestamp = Date.now()
    const finalElapsed = startTime !== null ? timestamp - startTime : elapsedMs
    const elapsedSecondsFinal = Math.max(finalElapsed / 1000, 0.01)
    const minutes = Math.max(elapsedSecondsFinal / 60, 1 / 60)
    const rawWpm = (metrics.correct / 5) / minutes
    const finalWpm = Number(rawWpm.toFixed(2))
    const finalAccuracy = Number(metrics.accuracy.toFixed(2))

    const session: TypingSession = {
      id: `session-${timestamp}`,
      language,
      snippetId: snippet.id,
      snippetTitle: snippet.title,
      wpm: finalWpm,
      accuracy: finalAccuracy,
      errors: metrics.mismatches,
      characters: snippetLength,
      durationSeconds: Number(elapsedSecondsFinal.toFixed(2)),
      timestamp,
    }

    setIsRunning(false)
    setStartTime(null)
    setElapsedMs(finalElapsed)
    setHasFinished(true)
    setSpeedSeries((previous) => {
      const nextPoint = { time: elapsedSecondsFinal, wpm: finalWpm }
      if (!previous.length) {
        return [{ time: 0, wpm: 0 }, nextPoint]
      }
      const lastPoint = previous[previous.length - 1]
      if (Math.abs(lastPoint.time - nextPoint.time) < 0.05) {
        const clone = [...previous]
        clone[clone.length - 1] = nextPoint
        return clone
      }
      const trimmed = previous.length > 300 ? previous.slice(previous.length - 300) : previous
      return [...trimmed, nextPoint]
    })
    setHistory((previous) => {
      const next = [session, ...previous].slice(0, 500)
      storeHistory(next)
      return next
    })
    setLastSession(session)
    setShowResultsModal(true)

    let newlyUnlocked: string | null = null
    if (unlockedTierIndex < TOURNAMENT_TIERS.length - 1) {
      const nextTierInfo = TOURNAMENT_TIERS[unlockedTierIndex + 1]
      if (finalWpm >= nextTierInfo.requiredWpm) {
        const nextIndex = unlockedTierIndex + 1
        setUnlockedTierIndex(nextIndex)
        newlyUnlocked = nextTierInfo.label
        if (typeof window !== "undefined") {
          window.localStorage.setItem(UNLOCK_STORAGE_KEY, String(nextIndex))
        }
      }
    }

    if (typeof window !== "undefined") {
      const currentHits = Number.parseInt(window.localStorage.getItem(HITS_STORAGE_KEY) ?? "0", 10)
      const updatedHits = Number.isNaN(currentHits) ? 1 : currentHits + 1
      window.localStorage.setItem(HITS_STORAGE_KEY, String(updatedHits))
      setSiteHits(updatedHits)
    }
    setUnlockNotice(newlyUnlocked)
  }, [elapsedMs, hasFinished, language, metrics.accuracy, metrics.correct, metrics.mismatches, snippet.id, snippet.title, snippetLength, startTime, unlockedTierIndex])

  useEffect(() => {
    if (!isRunning || startTime === null || hasFinished) {
      return
    }
    const interval = window.setInterval(() => {
      setElapsedMs(Date.now() - startTime)
    }, 200)
    return () => window.clearInterval(interval)
  }, [hasFinished, isRunning, startTime])

  useEffect(() => {
    if (!isRunning || startTime === null || hasFinished) {
      return
    }
    const now = Date.now()
    const elapsed = (now - startTime) / 1000
    const minutes = Math.max(elapsed / 60, 1 / 60)
    const liveWpm = (metrics.correct / 5) / minutes
    setSpeedSeries((previous) => {
      const nextPoint = { time: elapsed, wpm: liveWpm }
      if (!previous.length) {
        return [{ time: 0, wpm: 0 }, nextPoint]
      }
      const last = previous[previous.length - 1]
      if (elapsed - last.time < 0.2) {
        const clone = [...previous]
        clone[clone.length - 1] = nextPoint
        return clone
      }
      const trimmed = previous.length > 300 ? previous.slice(previous.length - 300) : previous
      return [...trimmed, nextPoint]
    })
  }, [hasFinished, isRunning, metrics.correct, startTime])

  useEffect(() => {
    if (remainingCharacters === 0 && !hasFinished && snippetLength > 0) {
      completeSession()
    }
  }, [completeSession, hasFinished, remainingCharacters, snippetLength])

  const snippetPreview = useMemo(() => {
    const characters = snippet.code.split("")
    return characters.map((char, index) => {
      const typedChar = userInput[index]
      const isTyped = index < userInput.length
      const isCorrect = typedChar === char
      const isCurrent = index === userInput.length && !hasFinished

      const className = cn(
        "inline-block rounded px-0.5 py-0.5",
        isTyped && isCorrect && "bg-primary/15 text-primary",
        isTyped && !isCorrect && "bg-destructive/30 text-destructive",
        !isTyped && !isCurrent && "text-muted-foreground",
        isCurrent && "border-b-2 border-primary text-primary",
      )

      const setCaret = isCurrent
        ? (node: HTMLSpanElement | null) => {
            caretRef.current = node
          }
        : undefined

      if (char === "\n") {
        return (
          <span key={`nl-${index}`} ref={setCaret} className="block w-full">
            {char}
          </span>
        )
      }

      return (
        <span key={index} ref={setCaret} className={className}>
          {char === " " ? " " : char}
        </span>
      )
    })
  }, [snippet, userInput, hasFinished])

  useEffect(() => {
    if (hasFinished) {
      return
    }
    const target = caretRef.current
    if (target) {
      target.scrollIntoView({ block: "nearest", inline: "nearest", behavior: "smooth" })
    }
  }, [userInput.length, snippet.id, language, codeScale, codePaneWidth, hasFinished])

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-background/90 pb-20">
      <main className="mx-auto flex w-full max-w-6xl flex-col gap-10 px-4 pt-12 sm:px-6 lg:px-8">
        <header className="rounded-2xl border bg-background/70 px-5 py-4 shadow-sm">
          <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex items-center gap-4">
              <div className="flex h-14 w-14 items-center justify-center overflow-hidden rounded-xl bg-muted shadow">
                <img src="/logo.svg" alt="KeySensei logo" className="h-12 w-12" />
              </div>
              <div className="flex flex-col gap-1">
                <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">KeySensei</h1>
                <p className="max-w-xl text-sm text-muted-foreground">
                  Train your typing instincts with curated code snippets and level-based progression.
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3 text-xs sm:text-sm">
              <span className="rounded-full bg-muted px-3 py-1 font-medium text-muted-foreground">
                Tier {tierDetails.index + 1}: {tierDetails.label}
              </span>
              <Button
                variant="ghost"
                size="icon"
                onClick={handleToggleTheme}
                aria-label={isDarkMode ? "Switch to light mode" : "Switch to dark mode"}
              >
                {isDarkMode ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
              </Button>
            </div>
          </div>
        </header>

        <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as "practice" | "history")}>
          <TabsList className="mx-auto mb-6 grid h-auto w-full max-w-md grid-cols-2 rounded-full bg-muted/60 p-1">
            <TabsTrigger value="practice" className="rounded-full px-4 py-2 text-sm">
              Practice
            </TabsTrigger>
            <TabsTrigger value="history" className="rounded-full px-4 py-2 text-sm">
              History
            </TabsTrigger>
          </TabsList>

          <TabsContent value="practice" className="mt-0">
            <div className={practiceLayoutClass} style={practiceLayoutStyle}>
              <Card className="shadow-sm">
                <CardHeader className="gap-2">
                  <CardTitle className="text-xl sm:text-2xl">Typing arena</CardTitle>
                  <CardDescription>Type the snippet exactly as shown. The round ends once every character matches.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                    <div className="flex items-center gap-2">
                      <Select value={language} onValueChange={handleLanguageChange}>
                        <SelectTrigger className="w-40" aria-label="Choose language">
                          <SelectValue placeholder="Language" />
                        </SelectTrigger>
                        <SelectContent>
                          {LANGUAGE_OPTIONS.map((option) => (
                            <SelectItem key={option.value} value={option.value}>
                              {option.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <Button variant="outline" size="sm" onClick={handleNewSnippet} className="gap-1 text-xs">
                        <RefreshCw className="h-3.5 w-3.5" />
                        New snippet
                      </Button>
                    </div>
                    <div className="flex items-center gap-3 text-xs text-muted-foreground">
                      <span>Progress {Math.round(completion * 100)}%</span>
                      <span className="hidden sm:inline">•</span>
                      <span>Next unlock ≥ {nextTier ? `${nextTier.requiredWpm} WPM` : "–"}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => adjustCodeScale(-CODE_SCALE_STEP)}
                        aria-label="Zoom out"
                        disabled={codeScale <= CODE_SCALE_MIN + 0.001}
                      >
                        <ZoomOut className="h-4 w-4" />
                      </Button>
                      <span className="min-w-[3rem] text-center text-xs font-semibold text-muted-foreground">
                        {Math.round(codeScale * 100)}%
                      </span>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => adjustCodeScale(CODE_SCALE_STEP)}
                        aria-label="Zoom in"
                        disabled={codeScale >= CODE_SCALE_MAX - 0.001}
                      >
                        <ZoomIn className="h-4 w-4" />
                      </Button>
                      <Button variant="ghost" size="sm" className="text-xs" onClick={resetCodeScale}>
                        Reset
                      </Button>
                    </div>
                  </div>

                  <div
                    ref={codePaneRef}
                    className="group relative rounded-lg border bg-muted/30"
                    style={codePaneStyle}
                  >
                    <div className="space-y-4 p-4">
                      <ScrollArea className="h-[260px] rounded-md bg-background/60 p-4">
                        <pre
                          className="whitespace-pre font-mono leading-relaxed"
                          style={{ fontSize: `${codeScale}rem`, lineHeight: codeScale > 1 ? 1.7 : 1.55 }}
                        >
                          {snippetPreview}
                        </pre>
                      </ScrollArea>

                      <Textarea
                        value={userInput}
                        onChange={handleTyping}
                        onKeyDown={handleKeyDown}
                        onPaste={blockClipboardEvent}
                        onCopy={blockClipboardEvent}
                        onCut={blockClipboardEvent}
                        disabled={hasFinished}
                        spellCheck={false}
                        autoCapitalize="off"
                        autoCorrect="off"
                        autoComplete="off"
                        className="min-h-[160px] w-full font-mono transition-all"
                        style={{ fontSize: `${codeScale}rem`, lineHeight: codeScale > 1 ? 1.5 : 1.4 }}
                        placeholder="Start typing here..."
                      />

                      <p className="text-xs text-muted-foreground">
                        Tip: tap <kbd>Tab</kbd> for indentation and drag the right edge to adjust the lane width.
                      </p>
                    </div>
                    <div
                      role="presentation"
                      onMouseDown={handleResizeMouseDown}
                      className="absolute top-0 right-0 flex h-full w-3 cursor-col-resize select-none items-center justify-center"
                    >
                      <span className="h-16 w-1 rounded-full bg-primary/40 transition-opacity group-hover:opacity-100" />
                    </div>
                  </div>
                  <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span>{remainingCharacters} characters remaining</span>
                    <span>{snippetLength} total · {snippet.code.split("\n").length} lines</span>
                  </div>
                </CardContent>
              </Card>

              <Card className="shadow-sm">
                <CardContent className="space-y-5 p-4">
                  <div className="grid gap-3 text-sm sm:grid-cols-4">
                    <div>
                      <p className="text-muted-foreground">WPM</p>
                      <p className="text-xl font-semibold text-foreground">{formatNumber(wpm, 1)}</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Accuracy</p>
                      <p className="text-xl font-semibold text-foreground">{formatNumber(accuracy, 1)}%</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Errors</p>
                      <p className="text-xl font-semibold text-foreground">{errors}</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Elapsed</p>
                      <p className="text-xl font-semibold text-foreground">{formatDuration(elapsedSeconds)}</p>
                    </div>
                  </div>
                  <div className="rounded-lg border bg-muted/30 p-3">
                    <SpeedGraph data={speedSeries} />
                  </div>
                  <div className="rounded-lg bg-muted/20 p-3 text-xs text-muted-foreground">
                    <p className="font-medium text-foreground">Focus next</p>
                    <ul className="mt-2 space-y-1">
                      {improvementTips.slice(0, 3).map((tip, index) => (
                        <li key={index}>{tip}</li>
                      ))}
                    </ul>
                    <p className="mt-3 text-muted-foreground">
                      Personal best: {bestForLanguage ? `${formatNumber(bestForLanguage.wpm, 1)} WPM` : "-"}
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="history" className="mt-0">
            <div className="space-y-6">
              <Card className="shadow-sm">
                <CardHeader>
                  <CardTitle className="text-xl">Overview</CardTitle>
                  <CardDescription>Your recent sessions stay on this device.</CardDescription>
                </CardHeader>
                <CardContent>
                  {!historySummary ? (
                    <p className="text-sm text-muted-foreground">Complete a session to see your progress summary.</p>
                  ) : (
                    <div className="grid gap-3 sm:grid-cols-4">
                      <div>
                        <p className="text-xs uppercase text-muted-foreground">Total</p>
                        <p className="text-xl font-semibold">{historySummary.totalSessions}</p>
                      </div>
                      <div>
                        <p className="text-xs uppercase text-muted-foreground">Average WPM</p>
                        <p className="text-xl font-semibold">{formatNumber(historySummary.averageWpm, 1)}</p>
                      </div>
                      <div>
                        <p className="text-xs uppercase text-muted-foreground">Average accuracy</p>
                        <p className="text-xl font-semibold">{formatNumber(historySummary.averageAccuracy, 1)}%</p>
                      </div>
                      <div>
                        <p className="text-xs uppercase text-muted-foreground">Languages</p>
                        <p className="text-xl font-semibold">{historySummary.languagesPractised}</p>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card className="shadow-sm">
                <CardHeader>
                  <CardTitle className="text-xl">Recent sessions</CardTitle>
                </CardHeader>
                <CardContent>
                  {!history.length ? (
                    <div className="flex h-40 items-center justify-center rounded-lg border border-dashed text-sm text-muted-foreground">
                      No sessions yet.
                    </div>
                  ) : (
                    <ScrollArea className="max-h-[420px] pr-4">
                      <div className="space-y-3">
                        {history.map((entry) => (
                          <div key={entry.id} className="rounded-lg border bg-background/70 p-4">
                            <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
                              <div className="flex items-center gap-2">
                                <Badge variant="secondary">{languageLabel(entry.language)}</Badge>
                                <span className="font-medium text-foreground">{formatNumber(entry.wpm, 1)} WPM</span>
                              </div>
                              <span className="text-xs text-muted-foreground">{new Date(entry.timestamp).toLocaleString()}</span>
                            </div>
                            <p className="mt-2 text-xs text-muted-foreground">{entry.snippetTitle}</p>
                          </div>
                        ))}
                      </div>
                    </ScrollArea>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>


        {showResultsModal && lastSession && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4 py-8">
            <div className="w-full max-w-xl rounded-2xl border bg-background/95 p-6 shadow-2xl">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                <div>
                  <h2 className="text-2xl font-semibold">Round complete</h2>
                  <p className="mt-1 text-sm text-muted-foreground">
                    You cleared <span className="font-medium text-foreground">{lastSession.snippetTitle}</span> with
                    {" "}
                    {formatNumber(lastSession.wpm, 1)} WPM at {formatNumber(lastSession.accuracy, 1)}% accuracy.
                  </p>
                </div>
                <Badge variant="outline" className="gap-1 px-3 py-1 text-xs font-semibold uppercase tracking-wide">
                  <Trophy className="h-3.5 w-3.5 text-primary" /> {tierDetails.label}
                </Badge>
              </div>

              {unlockNotice && (
                <div className="mt-4 rounded-lg border border-primary bg-primary/10 px-3 py-2 text-sm font-semibold text-primary">
                  {unlockNotice} unlocked! New drills are now available.
                </div>
              )}

              <div className="mt-5 grid gap-4 sm:grid-cols-2">
                <div className="rounded-lg border bg-background/80 p-4 text-sm shadow-sm">
                  <span className="text-muted-foreground">Words per minute</span>
                  <p className="mt-1 text-2xl font-semibold">{formatNumber(lastSession.wpm, 1)}</p>
                </div>
                <div className="rounded-lg border bg-background/80 p-4 text-sm shadow-sm">
                  <span className="text-muted-foreground">Accuracy</span>
                  <p className="mt-1 text-2xl font-semibold">{formatNumber(lastSession.accuracy, 1)}%</p>
                </div>
                <div className="rounded-lg border bg-background/80 p-4 text-sm shadow-sm">
                  <span className="text-muted-foreground">Errors</span>
                  <p className="mt-1 text-2xl font-semibold">{lastSession.errors}</p>
                </div>
                <div className="rounded-lg border bg-background/80 p-4 text-sm shadow-sm">
                  <span className="text-muted-foreground">Duration</span>
                  <p className="mt-1 text-2xl font-semibold">{formatDuration(lastSession.durationSeconds)}</p>
                </div>
              </div>

              <div className="mt-5 rounded-lg border bg-muted/30 p-4 text-sm text-muted-foreground">
                <p>{tierDetails.description}</p>
                <p className="mt-2 text-xs">
                  {nextTier
                    ? `Next up: ${nextTier.label}. Complete more rounds to unlock the next bracket.`
                    : "You are at the summit. Keep replaying to sharpen every keystroke."}
                </p>
              </div>

              <div className="mt-6 flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
                <Button variant="outline" onClick={handleDismissResults}>
                  Thanks
                </Button>
                <Button onClick={handleNextRound} className="gap-2">
                  <RefreshCw className="h-4 w-4" />
                  Next round
                </Button>
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="mx-auto w-full max-w-6xl px-6 pb-12 text-sm text-muted-foreground sm:px-8 lg:px-10">
        <div className="rounded-2xl border bg-background/70 p-6 shadow-lg sm:p-8">
          <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="text-lg font-semibold text-foreground">KeySensei</p>
              <p className="mt-1 text-sm text-muted-foreground">
                Keep climbing the ladder—easy to hard, one legendary algorithm at a time.
              </p>
            </div>
            <div className="text-sm sm:text-right">
              <p className="text-muted-foreground">Total dojo hits</p>
              <p className="text-xl font-semibold text-foreground">{formattedSiteHits}</p>
            </div>
          </div>
          <div className="mt-6 border-t border-border pt-4 text-center text-xs font-medium uppercase tracking-[0.3em] text-muted-foreground sm:pt-6">
            made with ❤️ by Ankit Adhikari
          </div>
        </div>
      </footer>
    </div>
  )
}

export default App
