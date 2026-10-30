import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import * as d3Force from 'd3-force';
import * as d3Zoom from 'd3-zoom';
import * as d3Selection from 'd3-selection';
import {
  Search,
  Maximize2,
  RotateCcw,
  ZoomIn,
  ZoomOut,
  Layers,
  FileText,
  Database,
  X,
  Loader2,
  MessageSquare,
  AlertCircle,
  RefreshCw,
  Quote
} from 'lucide-react';

export type EntityCategory = 'document' | 'concept' | 'technology' | 'process' | 'component' | 'chunk';

export interface GraphNode extends d3Force.SimulationNodeDatum {
  id: string;
  label: string;
  type: EntityCategory;
  document: string;
  size: number;
  color: string;
  excerpt?: string;
  full_text?: string;
  page?: number;
  frequency?: number;
  page_count?: number;
  chunk_count?: number;
}

export interface GraphEdge extends d3Force.SimulationLinkDatum<GraphNode> {
  id: string;
  source: string | GraphNode;
  target: string | GraphNode;
  relationship: string;
  predicate?: string;
  evidence?: string;
  page?: number;
  chunk_id?: string;
  document?: string;
  color?: string;
  weight?: number;
}

interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  stats: {
    total_nodes: number;
    documents_count: number;
    concepts_count: number;
    chunks_count: number;
    total_edges: number;
  };
}

interface KnowledgeGraphProps {
  documents: string[];
  activeDoc: string | null;
  onSelectDoc: (doc: string) => void;
  onQueryInChat: (query: string) => void;
}

const CATEGORY_COLORS: Record<EntityCategory, string> = {
  document: '#06b6d4',   // Cyan
  technology: '#38bdf8', // Sky Blue
  concept: '#818cf8',    // Indigo
  process: '#ec4899',    // Pink
  component: '#a855f7',  // Violet
  chunk: '#64748b',      // Slate
};

export const KnowledgeGraph: React.FC<KnowledgeGraphProps> = ({
  documents,
  activeDoc,
  onSelectDoc,
  onQueryInChat,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [refreshing, setRefreshing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const [selectedDocFilter, setSelectedDocFilter] = useState<string>(activeDoc || 'all');
  const [showDocuments, setShowDocuments] = useState<boolean>(true);
  const [showEntities, setShowEntities] = useState<boolean>(true);
  const [showChunks, setShowChunks] = useState<boolean>(true);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [depthFilter, setDepthFilter] = useState<number>(1);

  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const selectedNodeRef = useRef<GraphNode | null>(null);
  const hoveredNodeRef = useRef<GraphNode | null>(null);

  useEffect(() => { selectedNodeRef.current = selectedNode; }, [selectedNode]);
  useEffect(() => { hoveredNodeRef.current = hoveredNode; }, [hoveredNode]);

  const simulationRef = useRef<d3Force.Simulation<GraphNode, GraphEdge> | null>(null);
  const zoomBehaviorRef = useRef<d3Zoom.ZoomBehavior<HTMLCanvasElement, unknown> | null>(null);
  const transformRef = useRef<d3Zoom.ZoomTransform>(d3Zoom.zoomIdentity);
  const neighborMapRef = useRef<Map<string, Set<string>>>(new Map());

  const fetchGraph = useCallback(async (docName?: string, forceRefresh: boolean = false) => {
    if (forceRefresh) setRefreshing(true);
    else setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams();
      if (docName && docName !== 'all') {
        params.set('document_name', docName);
      }
      if (forceRefresh) {
        params.set('force_refresh', 'true');
      }

      const queryString = params.toString();
      const url = queryString ? `/api/graph?${queryString}` : '/api/graph';

      const res = await fetch(url);
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `HTTP error ${res.status}`);
      }
      const data: GraphData = await res.json();
      setGraphData(data);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to fetch knowledge graph data';
      setError(msg);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    fetchGraph(selectedDocFilter, false);
  }, [fetchGraph, selectedDocFilter]);

  const handleRefreshGraph = () => {
    fetchGraph(selectedDocFilter, true);
  };

  const filteredData = useMemo(() => {
    if (!graphData) return { nodes: [], edges: [] };

    const allowedTypes = new Set<string>();
    if (showDocuments) allowedTypes.add('document');
    if (showEntities) {
      allowedTypes.add('concept');
      allowedTypes.add('technology');
      allowedTypes.add('process');
      allowedTypes.add('component');
    }
    if (showChunks) allowedTypes.add('chunk');

    let visibleNodes = graphData.nodes.filter((n) => allowedTypes.has(n.type));
    const visibleNodeIdSet = new Set(visibleNodes.map((n) => n.id));

    let visibleEdges = graphData.edges.filter((e) => {
      const sId = typeof e.source === 'object' ? (e.source as GraphNode).id : e.source;
      const tId = typeof e.target === 'object' ? (e.target as GraphNode).id : e.target;
      return visibleNodeIdSet.has(sId) && visibleNodeIdSet.has(tId);
    });

    if (selectedNode && depthFilter < 99) {
      const currentLevelNodes = new Set<string>([selectedNode.id]);
      const visitedNodes = new Set<string>([selectedNode.id]);
      for (let level = 0; level < depthFilter; level++) {
        const nextLevelNodes = new Set<string>();
        visibleEdges.forEach((e) => {
          const sId = typeof e.source === 'object' ? (e.source as GraphNode).id : e.source;
          const tId = typeof e.target === 'object' ? (e.target as GraphNode).id : e.target;
          if (currentLevelNodes.has(sId) && !visitedNodes.has(tId)) {
            nextLevelNodes.add(tId);
            visitedNodes.add(tId);
          }
          if (currentLevelNodes.has(tId) && !visitedNodes.has(sId)) {
            nextLevelNodes.add(sId);
            visitedNodes.add(sId);
          }
        });
        currentLevelNodes.clear();
        nextLevelNodes.forEach((id) => currentLevelNodes.add(id));
      }
      visibleNodes = visibleNodes.filter((n) => visitedNodes.has(n.id));
      const subSet = new Set(visibleNodes.map((n) => n.id));
      visibleEdges = visibleEdges.filter((e) => {
        const sId = typeof e.source === 'object' ? (e.source as GraphNode).id : e.source;
        const tId = typeof e.target === 'object' ? (e.target as GraphNode).id : e.target;
        return subSet.has(sId) && subSet.has(tId);
      });
    }

    return { nodes: visibleNodes, edges: visibleEdges };
  }, [graphData, showDocuments, showEntities, showChunks, selectedNode, depthFilter]);

  const neighborMap = useMemo(() => {
    const map = new Map<string, Set<string>>();
    filteredData.edges.forEach((e) => {
      const sId = typeof e.source === 'object' ? (e.source as GraphNode).id : e.source;
      const tId = typeof e.target === 'object' ? (e.target as GraphNode).id : e.target;
      if (!map.has(sId)) map.set(sId, new Set());
      if (!map.has(tId)) map.set(tId, new Set());
      map.get(sId)!.add(tId);
      map.get(tId)!.add(sId);
    });
    return map;
  }, [filteredData.edges]);

  useEffect(() => { neighborMapRef.current = neighborMap; }, [neighborMap]);

  // Main Canvas D3 Force Simulation & Renderer
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || filteredData.nodes.length === 0) return;

    const width = container.clientWidth;
    const height = container.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    const simNodes: GraphNode[] = filteredData.nodes.map((n) => ({ ...n }));
    const simEdges: GraphEdge[] = filteredData.edges.map((e) => ({ ...e }));

    const simulation = d3Force
      .forceSimulation<GraphNode, GraphEdge>(simNodes)
      .force(
        'link',
        d3Force
          .forceLink<GraphNode, GraphEdge>(simEdges)
          .id((d) => d.id)
          .distance((d) => (d.relationship === 'contains' ? 55 : 90))
          .strength(0.4)
      )
      .force('charge', d3Force.forceManyBody().strength(-140).distanceMax(450))
      .force('center', d3Force.forceCenter(width / 2, height / 2).strength(0.05))
      .force('collide', d3Force.forceCollide<GraphNode>().radius((d) => d.size + 10).iterations(2))
      .alphaDecay(0.025);

    simulationRef.current = simulation;

    const render = () => {
      const selNode = selectedNodeRef.current;
      const hovNode = hoveredNodeRef.current;
      const nbrMap = neighborMapRef.current;
      const transform = transformRef.current;

      ctx.save();
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#06070a';
      ctx.fillRect(0, 0, width, height);
      ctx.translate(transform.x, transform.y);
      ctx.scale(transform.k, transform.k);

      const activeFocusId = selNode?.id || hovNode?.id || null;
      const neighbors = activeFocusId ? nbrMap.get(activeFocusId) || new Set() : null;

      // 1. Draw Edges
      simEdges.forEach((edge) => {
        const source = edge.source as GraphNode;
        const target = edge.target as GraphNode;
        if (!source.x || !source.y || !target.x || !target.y) return;

        const isConnected =
          activeFocusId &&
          (source.id === activeFocusId ||
            target.id === activeFocusId ||
            (neighbors && neighbors.has(source.id) && neighbors.has(target.id)));

        ctx.beginPath();
        ctx.moveTo(source.x, source.y);
        ctx.lineTo(target.x, target.y);

        if (activeFocusId) {
          if (isConnected) {
            ctx.strokeStyle =
              edge.relationship === 'contains'
                ? 'rgba(6, 182, 212, 0.8)'
                : edge.evidence
                ? 'rgba(236, 72, 153, 0.9)'
                : 'rgba(129, 140, 248, 0.8)';
            ctx.lineWidth = 1.8 / transform.k;
          } else {
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
            ctx.lineWidth = 0.5 / transform.k;
          }
        } else {
          ctx.strokeStyle =
            edge.relationship === 'contains'
              ? 'rgba(6, 182, 212, 0.22)'
              : edge.evidence
              ? 'rgba(236, 72, 153, 0.35)'
              : 'rgba(255, 255, 255, 0.1)';
          ctx.lineWidth = 0.8 / transform.k;
        }
        ctx.stroke();

        // Draw predicate label on connected edges or when zoomed in
        const showPredicate = (isConnected || transform.k > 1.8) && edge.predicate;
        if (showPredicate && activeFocusId) {
          const midX = (source.x + target.x) / 2;
          const midY = (source.y + target.y) / 2;
          ctx.font = `${Math.max(9, Math.min(11, 10 / transform.k))}px Inter, sans-serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          const predLabel = edge.predicate || edge.relationship;
          const textMetrics = ctx.measureText(predLabel);
          ctx.fillStyle = 'rgba(10, 10, 15, 0.85)';
          ctx.fillRect(
            midX - textMetrics.width / 2 - 3,
            midY - 7,
            textMetrics.width + 6,
            14
          );
          ctx.fillStyle = isConnected ? '#f472b6' : 'rgba(255, 255, 255, 0.6)';
          ctx.fillText(predLabel, midX, midY);
        }
      });

      // 2. Draw Nodes
      simNodes.forEach((node) => {
        if (!node.x || !node.y) return;
        const isSelected = selNode?.id === node.id;
        const isHovered = hovNode?.id === node.id;
        const isNeighbor = neighbors?.has(node.id);
        const isFocused = !activeFocusId || isSelected || isHovered || isNeighbor;
        const alpha = isFocused ? 1.0 : 0.12;
        const radius = node.size * (isSelected ? 1.3 : isHovered ? 1.15 : 1.0);

        // Glow effect
        if (isFocused && (node.type === 'document' || node.type !== 'chunk' || isSelected)) {
          const glowGrad = ctx.createRadialGradient(node.x, node.y, radius * 0.4, node.x, node.y, radius * 2.5);
          glowGrad.addColorStop(
            0,
            node.type === 'document'
              ? 'rgba(6, 182, 212, 0.4)'
              : node.type === 'technology'
              ? 'rgba(56, 189, 248, 0.35)'
              : node.type === 'process'
              ? 'rgba(236, 72, 153, 0.35)'
              : 'rgba(129, 140, 248, 0.35)'
          );
          glowGrad.addColorStop(1, 'transparent');
          ctx.fillStyle = glowGrad;
          ctx.beginPath();
          ctx.arc(node.x, node.y, radius * 2.5, 0, Math.PI * 2);
          ctx.fill();
        }

        // Selection ring
        if (isSelected) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, radius + 4 / transform.k, 0, Math.PI * 2);
          ctx.strokeStyle = '#38bdf8';
          ctx.lineWidth = 2.5 / transform.k;
          ctx.stroke();
        }

        // Node Circle
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = node.color || CATEGORY_COLORS[node.type] || '#818cf8';
        ctx.globalAlpha = alpha;
        ctx.fill();
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.7)';
        ctx.lineWidth = 1 / transform.k;
        ctx.stroke();
        ctx.globalAlpha = 1.0;

        // Label Rendering
        const shouldShowLabel =
          isSelected ||
          isHovered ||
          node.type === 'document' ||
          (transform.k > 1.1 && node.type !== 'chunk') ||
          transform.k > 2.0;

        if (shouldShowLabel && isFocused) {
          ctx.font = `${Math.max(10, Math.min(13, 11 / transform.k))}px Inter, sans-serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'top';
          const labelText = node.label.length > 24 ? `${node.label.slice(0, 22)}...` : node.label;
          const textMetrics = ctx.measureText(labelText);
          const bgPadding = 2;
          ctx.fillStyle = 'rgba(5, 5, 8, 0.85)';
          ctx.fillRect(
            node.x - textMetrics.width / 2 - bgPadding,
            node.y + radius + 4 - bgPadding,
            textMetrics.width + bgPadding * 2,
            14 + bgPadding * 2
          );

          ctx.fillStyle =
            node.type === 'document'
              ? '#67e8f9'
              : node.type === 'technology'
              ? '#7dd3fc'
              : node.type === 'process'
              ? '#f472b6'
              : node.type === 'component'
              ? '#c084fc'
              : node.type === 'concept'
              ? '#c7d2fe'
              : '#94a3b8';

          ctx.fillText(labelText, node.x, node.y + radius + 4);
        }
      });

      ctx.restore();
    };

    simulation.on('tick', render);

    const zoomBehavior = d3Zoom
      .zoom<HTMLCanvasElement, unknown>()
      .scaleExtent([0.15, 8])
      .on('zoom', (event) => {
        transformRef.current = event.transform;
        render();
      });

    zoomBehaviorRef.current = zoomBehavior;
    d3Selection.select(canvas).call(zoomBehavior);

    let draggedNode: GraphNode | null = null;

    const findNodeAtPoint = (px: number, py: number): GraphNode | null => {
      const transform = transformRef.current;
      const x = (px - transform.x) / transform.k;
      const y = (py - transform.y) / transform.k;
      for (const node of simNodes) {
        if (node.x === undefined || node.y === undefined) continue;
        if (Math.hypot(node.x - x, node.y - y) <= node.size + 8) return node;
      }
      return null;
    };

    const handleMouseDown = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const clicked = findNodeAtPoint(e.clientX - rect.left, e.clientY - rect.top);
      if (clicked) {
        draggedNode = clicked;
        clicked.fx = clicked.x;
        clicked.fy = clicked.y;
        simulation.alphaTarget(0.3).restart();
        selectedNodeRef.current = clicked;
        setSelectedNode(clicked);
      }
    };

    const handleMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      if (draggedNode) {
        const transform = transformRef.current;
        draggedNode.fx = (px - transform.x) / transform.k;
        draggedNode.fy = (py - transform.y) / transform.k;
      } else {
        const hovered = findNodeAtPoint(px, py);
        canvas.style.cursor = hovered ? 'pointer' : 'grab';
        const prevId = hoveredNodeRef.current?.id ?? null;
        const newId = hovered?.id ?? null;
        if (prevId !== newId) {
          hoveredNodeRef.current = hovered;
          setHoveredNode(hovered);
          render();
        }
      }
    };

    const handleMouseUp = () => {
      if (draggedNode) {
        draggedNode.fx = null;
        draggedNode.fy = null;
        draggedNode = null;
        simulation.alphaTarget(0);
      }
    };

    canvas.addEventListener('mousedown', handleMouseDown);
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      simulation.stop();
      canvas.removeEventListener('mousedown', handleMouseDown);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [filteredData, neighborMap]);

  const handleFitView = () => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || !simulationRef.current || !zoomBehaviorRef.current) return;
    const nodes = simulationRef.current.nodes();
    if (nodes.length === 0) return;

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    nodes.forEach((n) => {
      if (n.x === undefined || n.y === undefined) return;
      if (n.x < minX) minX = n.x;
      if (n.x > maxX) maxX = n.x;
      if (n.y < minY) minY = n.y;
      if (n.y > maxY) maxY = n.y;
    });

    const width = container.clientWidth;
    const height = container.clientHeight;
    const dx = maxX - minX || 1;
    const dy = maxY - minY || 1;
    const midX = (minX + maxX) / 2;
    const midY = (minY + maxY) / 2;
    const scale = Math.min(2.5, Math.max(0.25, 0.85 / Math.max(dx / width, dy / height)));
    const translate = [width / 2 - scale * midX, height / 2 - scale * midY];

    zoomBehaviorRef.current.transform(
      d3Selection.select(canvas),
      d3Zoom.zoomIdentity.translate(translate[0], translate[1]).scale(scale)
    );
  };

  const handleReset = () => {
    setSelectedNode(null);
    selectedNodeRef.current = null;
    setHoveredNode(null);
    hoveredNodeRef.current = null;
    setSearchQuery('');
    setDepthFilter(1);
    if (simulationRef.current) simulationRef.current.alpha(1).restart();
    handleFitView();
  };

  const handleZoomIn = () => {
    if (canvasRef.current && zoomBehaviorRef.current) {
      zoomBehaviorRef.current.scaleBy(d3Selection.select(canvasRef.current), 1.3);
    }
  };

  const handleZoomOut = () => {
    if (canvasRef.current && zoomBehaviorRef.current) {
      zoomBehaviorRef.current.scaleBy(d3Selection.select(canvasRef.current), 0.77);
    }
  };

  const searchResults = useMemo(() => {
    if (!searchQuery.trim() || !graphData) return [];
    const query = searchQuery.toLowerCase().trim();
    return graphData.nodes
      .filter((n) => n.label.toLowerCase().includes(query) || (n.excerpt && n.excerpt.toLowerCase().includes(query)))
      .slice(0, 8);
  }, [searchQuery, graphData]);

  const handleSelectSearchResult = (node: GraphNode) => {
    selectedNodeRef.current = node;
    setSelectedNode(node);
    setSearchQuery('');
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (canvas && container && zoomBehaviorRef.current && node.x !== undefined && node.y !== undefined) {
      const width = container.clientWidth;
      const height = container.clientHeight;
      const scale = 2.0;
      zoomBehaviorRef.current.transform(
        d3Selection.select(canvas),
        d3Zoom.zoomIdentity.translate(width / 2 - scale * node.x, height / 2 - scale * node.y).scale(scale)
      );
    }
  };

  // Connected Triples for selected node
  const connectedTriples = useMemo(() => {
    if (!selectedNode || !graphData) return [];
    const id = selectedNode.id;
    return graphData.edges
      .filter((e) => {
        const sId = typeof e.source === 'object' ? (e.source as GraphNode).id : e.source;
        const tId = typeof e.target === 'object' ? (e.target as GraphNode).id : e.target;
        return sId === id || tId === id;
      })
      .map((edge) => {
        const sId = typeof edge.source === 'object' ? (edge.source as GraphNode).id : edge.source;
        const tId = typeof edge.target === 'object' ? (edge.target as GraphNode).id : edge.target;
        const otherId = sId === id ? tId : sId;
        const otherNode = graphData.nodes.find((n) => n.id === otherId);
        const isOutgoing = sId === id;
        return {
          edge,
          otherNode,
          predicate: edge.predicate || edge.relationship,
          isOutgoing,
          evidence: edge.evidence,
          page: edge.page,
        };
      })
      .filter((item) => Boolean(item.otherNode));
  }, [selectedNode, graphData]);

  return (
    <div className="relative w-full h-[calc(100dvh-4rem)] bg-[#050507] overflow-hidden flex flex-col font-sans select-none">
      {/* Top Floating Control Bar */}
      <div className="absolute top-4 left-4 right-4 z-20 flex flex-wrap items-center justify-between gap-3 pointer-events-none">
        {/* Left: Document filter & Search */}
        <div className="flex items-center space-x-2.5 pointer-events-auto">
          <div className="relative flex items-center">
            <Database className="absolute left-3 h-3.5 w-3.5 text-cyan-400 pointer-events-none" />
            <select
              value={selectedDocFilter}
              onChange={(e) => {
                setSelectedDocFilter(e.target.value);
                if (e.target.value !== 'all') onSelectDoc(e.target.value);
              }}
              className="rounded-2xl bg-[#111116]/85 backdrop-blur-xl border border-white/10 pl-8 pr-7 py-2 text-xs font-medium text-white focus:outline-none focus:border-cyan-400/50 shadow-xl cursor-pointer appearance-none"
            >
              <option value="all" className="bg-[#121216] text-white">
                All Indexed Documents ({documents.length})
              </option>
              {documents.map((d) => (
                <option key={d} value={d} className="bg-[#121216] text-white">
                  {d}
                </option>
              ))}
            </select>
          </div>

          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-white/40" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search concepts, entities..."
              className="w-52 sm:w-64 rounded-2xl bg-[#111116]/85 backdrop-blur-xl border border-white/10 pl-8 pr-3 py-2 text-xs text-white placeholder-white/30 focus:outline-none focus:border-cyan-400/50 shadow-xl"
            />
            {searchResults.length > 0 && (
              <div className="absolute top-full left-0 mt-2 w-72 rounded-2xl bg-[#121218]/95 backdrop-blur-2xl border border-white/15 p-2 shadow-2xl space-y-1 z-30">
                <div className="text-[10px] font-mono uppercase text-white/40 px-2 py-1">
                  Matching Nodes ({searchResults.length})
                </div>
                {searchResults.map((item) => (
                  <button
                    key={item.id}
                    onClick={() => handleSelectSearchResult(item)}
                    className="w-full text-left px-2.5 py-1.5 rounded-xl hover:bg-white/10 flex items-center justify-between transition-colors group"
                  >
                    <div className="truncate pr-2">
                      <span className="text-xs text-white font-medium group-hover:text-cyan-300">
                        {item.label}
                      </span>
                      <p className="text-[10px] text-white/40 truncate">{item.document}</p>
                    </div>
                    <span
                      className="text-[9px] font-mono uppercase px-1.5 py-0.5 rounded"
                      style={{
                        backgroundColor: `${CATEGORY_COLORS[item.type] || '#818cf8'}22`,
                        color: CATEGORY_COLORS[item.type] || '#818cf8',
                      }}
                    >
                      {item.type}
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Right: Category filters & Zoom Controls */}
        <div className="flex items-center space-x-2 pointer-events-auto">
          {/* Entity category toggles */}
          <div className="hidden md:flex items-center space-x-1 rounded-2xl bg-[#111116]/85 backdrop-blur-xl border border-white/10 p-1 shadow-xl text-xs">
            <button
              onClick={() => setShowDocuments(!showDocuments)}
              className={`flex items-center space-x-1 px-2.5 py-1 rounded-xl transition-all ${
                showDocuments
                  ? 'bg-cyan-500/20 text-cyan-300 font-medium'
                  : 'text-white/40 hover:text-white'
              }`}
            >
              <span className="h-1.5 w-1.5 rounded-full bg-cyan-400" />
              <span>Docs</span>
            </button>
            <button
              onClick={() => setShowEntities(!showEntities)}
              className={`flex items-center space-x-1 px-2.5 py-1 rounded-xl transition-all ${
                showEntities
                  ? 'bg-indigo-500/20 text-indigo-300 font-medium'
                  : 'text-white/40 hover:text-white'
              }`}
            >
              <span className="h-1.5 w-1.5 rounded-full bg-indigo-400" />
              <span>Entities</span>
            </button>
            <button
              onClick={() => setShowChunks(!showChunks)}
              className={`flex items-center space-x-1 px-2.5 py-1 rounded-xl transition-all ${
                showChunks
                  ? 'bg-slate-500/20 text-slate-300 font-medium'
                  : 'text-white/40 hover:text-white'
              }`}
            >
              <span className="h-1.5 w-1.5 rounded-full bg-slate-400" />
              <span>Chunks</span>
            </button>
          </div>

          {/* Depth filter if node selected */}
          {selectedNode && (
            <div className="flex items-center space-x-1 rounded-2xl bg-[#111116]/85 backdrop-blur-xl border border-white/10 px-2 py-1 shadow-xl text-xs">
              <span className="text-[10px] font-mono text-white/40 px-1">Depth:</span>
              {[1, 2, 99].map((lvl) => (
                <button
                  key={lvl}
                  onClick={() => setDepthFilter(lvl)}
                  className={`px-2 py-0.5 rounded-lg text-xs font-mono transition-all ${
                    depthFilter === lvl
                      ? 'bg-white text-black font-semibold'
                      : 'text-white/50 hover:text-white'
                  }`}
                >
                  {lvl === 99 ? 'All' : `${lvl}x`}
                </button>
              ))}
            </div>
          )}

          {/* Re-extract graph button */}
          <button
            onClick={handleRefreshGraph}
            disabled={refreshing || loading}
            title="Re-extract knowledge graph with Groq"
            className="flex items-center space-x-1 px-2.5 py-1.5 rounded-2xl bg-[#111116]/85 hover:bg-white/10 border border-white/10 text-xs text-white/70 hover:text-white transition-all shadow-xl disabled:opacity-40"
          >
            <RefreshCw className={`h-3.5 w-3.5 text-cyan-400 ${refreshing ? 'animate-spin' : ''}`} />
            <span className="hidden sm:inline font-medium">Re-extract</span>
          </button>

          {/* Zoom & Fit controls */}
          <div className="flex items-center space-x-1 rounded-2xl bg-[#111116]/85 backdrop-blur-xl border border-white/10 p-1 shadow-xl">
            <button
              onClick={handleZoomIn}
              title="Zoom In"
              className="p-1.5 rounded-xl text-white/60 hover:text-white hover:bg-white/10 transition-colors"
            >
              <ZoomIn className="h-4 w-4" />
            </button>
            <button
              onClick={handleZoomOut}
              title="Zoom Out"
              className="p-1.5 rounded-xl text-white/60 hover:text-white hover:bg-white/10 transition-colors"
            >
              <ZoomOut className="h-4 w-4" />
            </button>
            <button
              onClick={handleFitView}
              title="Fit to View"
              className="p-1.5 rounded-xl text-white/60 hover:text-white hover:bg-white/10 transition-colors"
            >
              <Maximize2 className="h-4 w-4" />
            </button>
            <button
              onClick={handleReset}
              title="Reset Graph"
              className="p-1.5 rounded-xl text-white/60 hover:text-white hover:bg-white/10 transition-colors"
            >
              <RotateCcw className="h-4 w-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Graph Canvas Container */}
      <div ref={containerRef} className="flex-1 w-full h-full relative">
        <canvas ref={canvasRef} className="w-full h-full block cursor-grab active:cursor-grabbing" />

        {/* Loading Overlay */}
        {(loading || refreshing) && (
          <div className="absolute inset-0 bg-[#050507]/80 backdrop-blur-md flex flex-col items-center justify-center space-y-3 z-30">
            <Loader2 className="h-8 w-8 text-cyan-400 animate-spin" />
            <p className="text-xs font-mono tracking-wider text-white/70 uppercase">
              {refreshing
                ? 'Extracting Structured Triples with Groq...'
                : 'Loading Verified Knowledge Graph...'}
            </p>
          </div>
        )}

        {/* Error Notification */}
        {error && (
          <div className="absolute top-20 left-1/2 -translate-x-1/2 z-30 rounded-2xl bg-red-500/10 border border-red-500/30 px-4 py-2 flex items-center space-x-2 text-xs text-red-300 shadow-xl backdrop-blur-xl">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Empty State */}
        {!loading && filteredData.nodes.length === 0 && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-6 z-10 pointer-events-none">
            <div className="h-14 w-14 rounded-3xl bg-white/[0.04] border border-white/10 flex items-center justify-center text-white/40 mb-4 shadow-inner">
              <Layers className="h-7 w-7 text-cyan-400/60" />
            </div>
            <h3 className="text-base font-semibold text-white">No Knowledge Graph Available</h3>
            <p className="text-xs text-white/50 max-w-sm mt-1 leading-relaxed">
              Upload a PDF document to generate an interactive knowledge graph representing verified
              subject-predicate-object relationships.
            </p>
          </div>
        )}

        {/* Bottom Legend & Stats */}
        {graphData && (
          <div className="absolute bottom-4 left-4 z-10 rounded-2xl bg-[#111116]/85 backdrop-blur-xl border border-white/10 px-3.5 py-2 shadow-xl flex items-center space-x-3.5 text-[11px] text-white/60 font-mono">
            <div className="flex items-center space-x-1.5">
              <span className="h-2 w-2 rounded-full bg-cyan-400" />
              <span>Docs ({graphData.stats.documents_count})</span>
            </div>
            <div className="flex items-center space-x-1.5">
              <span className="h-2 w-2 rounded-full bg-indigo-400" />
              <span>Entities ({graphData.stats.concepts_count})</span>
            </div>
            <div className="flex items-center space-x-1.5">
              <span className="h-2 w-2 rounded-full bg-pink-400" />
              <span>Relations ({graphData.stats.total_edges})</span>
            </div>
          </div>
        )}
      </div>

      {/* Selected Node Details Drawer */}
      {selectedNode && (
        <div className="absolute top-0 right-0 bottom-0 w-full sm:w-[420px] bg-[#0e0e14]/95 backdrop-blur-2xl border-l border-white/10 p-6 shadow-2xl z-40 overflow-y-auto flex flex-col justify-between">
          <div>
            {/* Header */}
            <div className="flex items-center justify-between pb-4 border-b border-white/10">
              <div className="flex items-center space-x-2">
                <span
                  className="h-2.5 w-2.5 rounded-full"
                  style={{ backgroundColor: selectedNode.color || CATEGORY_COLORS[selectedNode.type] }}
                />
                <span className="text-xs font-mono uppercase tracking-wider text-white/50">
                  {selectedNode.type} Node
                </span>
                {selectedNode.frequency && (
                  <span className="text-[10px] font-mono bg-white/10 px-1.5 py-0.5 rounded text-white/70">
                    Freq: {selectedNode.frequency}
                  </span>
                )}
              </div>
              <button
                type="button"
                onClick={() => {
                  setSelectedNode(null);
                  selectedNodeRef.current = null;
                }}
                className="p-1 rounded-lg text-white/40 hover:text-white hover:bg-white/10 transition-colors"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            {/* Title & Document */}
            <div className="mt-4">
              <h3 className="text-lg font-semibold text-white leading-tight">
                {selectedNode.label}
              </h3>
              {selectedNode.document && (
                <div className="mt-2 flex items-center space-x-1.5 text-xs text-white/40">
                  <FileText className="h-3.5 w-3.5 text-cyan-400" />
                  <span className="font-mono truncate">{selectedNode.document}</span>
                </div>
              )}
            </div>

            {/* Context Definition */}
            <div className="mt-5">
              <span className="text-[11px] font-mono uppercase tracking-wider text-white/40 block mb-2">
                {selectedNode.type === 'chunk' ? 'Chunk Excerpt' : 'Context & Definition'}
              </span>
              <div className="rounded-2xl bg-white/[0.03] border border-white/10 p-3.5 text-xs text-white/80 leading-relaxed font-sans max-h-40 overflow-y-auto">
                {selectedNode.full_text || selectedNode.excerpt || 'No additional details.'}
              </div>
            </div>

            {/* Verified Relationships & Provenance Triples */}
            <div className="mt-6">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[11px] font-mono uppercase tracking-wider text-white/40">
                  Verified SPO Triples ({connectedTriples.length})
                </span>
              </div>
              <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
                {connectedTriples.length === 0 ? (
                  <p className="text-xs text-white/30 italic">No structured triples linked.</p>
                ) : (
                  connectedTriples.map((item, idx) => (
                    <div
                      key={idx}
                      className="rounded-2xl bg-white/[0.03] border border-white/5 p-3 text-xs space-y-1.5"
                    >
                      <div className="flex items-center justify-between">
                        <button
                          type="button"
                          onClick={() => item.otherNode && handleSelectSearchResult(item.otherNode)}
                          className="font-medium text-white hover:text-cyan-300 transition-colors flex items-center space-x-1 truncate"
                        >
                          <span className="text-[11px] font-mono text-pink-400 mr-1.5">
                            {item.isOutgoing ? '—[' : '←['}
                            {item.predicate}
                            {item.isOutgoing ? ']→' : ']—'}
                          </span>
                          <span className="truncate">{item.otherNode?.label}</span>
                        </button>
                        {item.page !== undefined && (
                          <span className="text-[10px] font-mono text-white/40 shrink-0 ml-1">
                            p.{item.page}
                          </span>
                        )}
                      </div>
                      {item.evidence && (
                        <div className="text-[11px] text-white/60 bg-white/[0.02] p-2 rounded-xl border border-white/[0.04] italic font-sans flex items-start space-x-1.5">
                          <Quote className="h-3 w-3 text-cyan-400/50 shrink-0 mt-0.5" />
                          <span className="line-clamp-2">&ldquo;{item.evidence}&rdquo;</span>
                        </div>
                      )}
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          {/* Action CTA */}
          <div className="pt-6 border-t border-white/10 mt-6">
            <button
              type="button"
              onClick={() =>
                onQueryInChat(
                  `Explain ${selectedNode.label} and how it relates to other concepts in ${
                    selectedNode.document || 'the document'
                  }`
                )
              }
              className="w-full flex items-center justify-center space-x-2 py-2.5 rounded-2xl bg-white text-black font-semibold text-xs hover:bg-white/90 shadow-lg transition-all active:scale-[0.98]"
            >
              <MessageSquare className="h-3.5 w-3.5" />
              <span>Ask Chatbot About This</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
