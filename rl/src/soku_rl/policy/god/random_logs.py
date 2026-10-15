"""Actor-local storage for the original my_rand.ai numeric replay logs."""


def install_random_logs(lua):
    return lua.execute(b"""
local files = {}
local native_open = io.open
io.open = function(name, mode)
  name = tostring(name)
  if name ~= '311000851' and name ~= '311000852' then
    return native_open(name, mode)
  end
  if mode ~= 'w' and mode ~= 'a+' and mode ~= 'r' then
    error('unsupported original random log mode: '..tostring(mode))
  end
  if mode == 'r' and files[name] == nil then return nil, 'missing actor random log' end
  if mode == 'w' or files[name] == nil then files[name] = '' end
  local position = 0
  local closed = false
  local handle = {}
  local function check()
    if closed then error('attempt to use a closed random log') end
  end
  function handle:write(...)
    check()
    if mode == 'r' then error('random log is read-only') end
    if mode == 'a+' then position = #files[name] end
    local text = ''
    for i = 1, select('#', ...) do text = text .. tostring(select(i, ...)) end
    files[name] = files[name]:sub(1, position) .. text .. files[name]:sub(position + #text + 1)
    position = position + #text
    return self
  end
  function handle:read(format)
    check()
    if format ~= '*n' then error('original random log requires numeric reads') end
    local first, last, token = files[name]:find('^%s*(%S+)', position + 1)
    if first == nil then return nil end
    local value = tonumber(token)
    if value ~= nil then position = last end
    return value
  end
  function handle:seek(whence, offset)
    check()
    local bases = {set=0, cur=position, ['end']=#files[name]}
    if bases[whence] == nil then error('invalid random log seek') end
    position = bases[whence] + (offset or 0)
    return position
  end
  function handle:flush() check(); return true end
  function handle:close() check(); closed = true; return true end
  return handle
end
return files
""")
