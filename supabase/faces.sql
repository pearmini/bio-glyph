-- BioGlyph community faces table (run in Supabase SQL Editor, bairui-studio)

create table public.faces (
  id uuid primary key default gen_random_uuid(),
  path jsonb not null check (jsonb_array_length(path) >= 2),
  browser_id text not null check (char_length(browser_id) >= 8),
  created_at timestamptz not null default now()
);

create index faces_created_at_idx on public.faces (created_at desc);

alter table public.faces enable row level security;

create policy "faces_select_public"
  on public.faces for select
  to anon, authenticated
  using (true);

create policy "faces_insert_public"
  on public.faces for insert
  to anon, authenticated
  with check (jsonb_array_length(path) >= 2);

create policy "faces_delete_own"
  on public.faces for delete
  to anon, authenticated
  using (
    browser_id = coalesce(
      (current_setting('request.headers', true)::json ->> 'x-browser-id'),
      ''
    )
  );

grant select, insert, delete on public.faces to anon, authenticated;
