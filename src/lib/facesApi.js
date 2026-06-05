import { getBrowserId } from "./browserId.js";
import { getSupabase, isSupabaseConfigured } from "./supabase.js";

function rowToFace(row) {
  return {
    id: row.id,
    path: row.path,
    createdAt: new Date(row.created_at).getTime(),
    browserId: row.browser_id,
    source: "community",
  };
}

export async function fetchCommunityFaces() {
  const supabase = getSupabase();
  if (!supabase) return [];

  const { data, error } = await supabase
    .from("faces")
    .select("id, path, browser_id, created_at")
    .order("created_at", { ascending: false });

  if (error) throw error;
  return (data ?? []).map(rowToFace);
}

export async function fetchCommunityFaceById(id) {
  const supabase = getSupabase();
  if (!supabase) return null;

  const { data, error } = await supabase
    .from("faces")
    .select("id, path, browser_id, created_at")
    .eq("id", id)
    .maybeSingle();

  if (error) throw error;
  return data ? rowToFace(data) : null;
}

export async function addCommunityFace(path) {
  const supabase = getSupabase();
  if (!supabase) {
    throw new Error("Archive is not connected yet. Please try again later.");
  }

  const { data, error } = await supabase
    .from("faces")
    .insert({ path, browser_id: getBrowserId() })
    .select("id, path, browser_id, created_at")
    .single();

  if (error) throw error;
  return rowToFace(data);
}

export async function deleteCommunityFace(id) {
  const supabase = getSupabase();
  if (!supabase) {
    throw new Error("Archive is not connected yet. Please try again later.");
  }

  const { error } = await supabase.from("faces").delete().eq("id", id);
  if (error) throw error;
}

export { isSupabaseConfigured };
