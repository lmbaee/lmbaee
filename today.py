"""
Reconstructed today.py

Purpose: Fetch GitHub statistics for a user (commits, stars, repos, followers, lines added/deleted)
and write those numbers into two SVG files (dark_mode.svg and light_mode.svg). Uses a local
cache to avoid re-scanning unchanged repositories. Designed to replicate and improve the
script extracted from the PDF you provided.

Notes:
 - Reads ACCESS_TOKEN and USER_NAME from the environment.
 - Writes/reads cache files to ./cache/<sha256(USER_NAME)>.txt
 - Updates dark_mode.svg and light_mode.svg in the current working directory.
 - If you want to test quickly, set DEBUG=1 in the environment to limit repo scans.

This is a cleaned, fixed, and more robust reconstruction based on the PDF you uploaded.
"""

import os
import sys
import time
import json
import hashlib
import logging
import datetime
from dateutil import relativedelta
import requests
from lxml import etree
from typing import Tuple, List, Dict, Any, Optional

# ----------------------------- Configuration ---------------------------------
# Environment variables (required): ACCESS_TOKEN, USER_NAME
# Optional environment variables used for debugging / testing:
# - DEBUG=1 : scans only a few repositories for quick local testing
# - CACHE_COMMENT_SIZE : integer, how many comment lines to preserve at top of cache (default 7)
# ----------------------------------------------------------------------------

# Basic logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("today_reconstructed")

ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN")
USER_NAME = os.environ.get("USER_NAME")
DEBUG = os.environ.get("DEBUG", "0") in ("1", "true", "True")
CACHE_COMMENT_SIZE = int(os.environ.get("CACHE_COMMENT_SIZE", "7"))

if not ACCESS_TOKEN or not USER_NAME:
    log.error("Both ACCESS_TOKEN and USER_NAME environment variables must be set.")
    log.error("Create a GitHub Personal Access Token and export it as ACCESS_TOKEN, and set YOUR github username as USER_NAME.")
    sys.exit(1)

# Use Bearer token (works with GitHub GraphQL). The original used 'token ' in the PDF; Bearer is fine.
HEADERS = {"Authorization": f"Bearer {ACCESS_TOKEN}"}

# Track query counts for diagnostics (same idea as the original script)
QUERY_COUNT: Dict[str, int] = {
    'user_getter': 0,
    'follower_getter': 0,
    'graph_repos_stars': 0,
    'recursive_loc': 0,
    'graph_commits': 0,
    'loc_query': 0,
}

#  Utilities 

def query_count(name: str) -> None:
    if name not in QUERY_COUNT:
        QUERY_COUNT[name] = 0
    QUERY_COUNT[name] += 1


def perf_counter(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - start


def format_plural(unit: int) -> str:
    return 's' if unit != 1 else ''


def daily_readme(birthday: datetime.datetime) -> str:
    diff = relativedelta.relativedelta(datetime.datetime.utcnow(), birthday)
    return f"{diff.years} year{format_plural(diff.years)}, {diff.months} month{format_plural(diff.months)}, {diff.days} day{format_plural(diff.days)}{(' 🎂' if (diff.months == 0 and diff.days == 0) else '')}"

#  HTTP / GraphQL 

def _post_graphql(payload: dict, retries: int = 5, backoff: float = 1.0) -> requests.Response:
    """Send a POST to GitHub GraphQL with simple retry/backoff for transient errors."""
    for attempt in range(1, retries + 1):
        resp = requests.post('https://api.github.com/graphql', json=payload, headers=HEADERS)
        if resp.status_code == 200:
            return resp
        # Rate limiting / abuse or transient server errors
        if resp.status_code in (429, 502, 503, 504, 403):
            wait = backoff * (2 ** (attempt - 1))
            log.warning(f"GraphQL request returned {resp.status_code}. Attempt {attempt}/{retries}. Waiting {wait}s and retrying...")
            time.sleep(wait)
            continue
        # Unrecoverable error
        log.error("GraphQL request failed: %s %s", resp.status_code, resp.text)
        resp.raise_for_status()
    # If we get here, raise with last response
    raise Exception("GraphQL request failed after retries")


def simple_request(func_name: str, query: str, variables: dict) -> requests.Response:
    query_count(func_name)
    payload = {'query': query, 'variables': variables}
    resp = _post_graphql(payload)
    return resp

#  GraphQL helpers 

def user_getter(username: str) -> Tuple[str, str]:
    """Return (owner_id, createdAt) for the username."""
    query = '''
    query($login: String!) {
      user(login: $login) {
        id
        createdAt
      }
    }
    '''
    r = simple_request('user_getter', query, {'login': username})
    data = r.json()['data']['user']
    return data['id'], data['createdAt']


def follower_getter(username: str) -> int:
    query = '''
    query($login: String!) {
      user(login: $login) { followers { totalCount } }
    }
    '''
    r = simple_request('follower_getter', query, {'login': username})
    return int(r.json()['data']['user']['followers']['totalCount'])


def graph_commits(start_date: str, end_date: str) -> int:
    # Note: the PDF version had stray spaces in field names (e.g. 'contributionsCollect ion').
    # This reconstruction uses the correct names: contributionsCollection -> contributionCalendar -> totalContributions.
    query = '''
    query($start_date: DateTime!, $end_date: DateTime!, $login: String!) {
      user(login: $login) {
        contributionsCollection(from: $start_date, to: $end_date) {
          contributionCalendar { totalContributions }
        }
      }
    }
    '''
    r = simple_request('graph_commits', query, {'start_date': start_date, 'end_date': end_date, 'login': USER_NAME})
    return int(r.json()['data']['user']['contributionsCollection']['contributionCalendar']['totalContributions'])


def graph_repos_stars(count_type: str, owner_affiliation: List[str], cursor: Optional[str] = None) -> Any:
    """Return number of repos (totalCount) or total stars (sum of stargazers) depending on count_type.
       owner_affiliation is a list like ['OWNER'] or ['OWNER','COLLABORATOR']
    """
    query = '''
    query ($ownerAffiliations: [RepositoryAffiliation], $login: String!, $cursor: String) {
      user(login: $login) {
        repositories(first: 100, after: $cursor, ownerAffiliations: $ownerAffiliations) {
          totalCount
          edges { node { ... on Repository { nameWithOwner stargazers { totalCount } } } }
          pageInfo { endCursor hasNextPage }
        }
      }
    }
    '''
    r = simple_request('graph_repos_stars', query, {'ownerAffiliations': owner_affiliation, 'login': USER_NAME, 'cursor': cursor})
    data = r.json()['data']['user']['repositories']
    if count_type == 'repos':
        return data['totalCount']
    elif count_type == 'stars':
        return sum(edge['node']['stargazers']['totalCount'] for edge in data['edges'])
    else:
        # For other uses, return raw page data
        return data

#  LOC counting 

def recursive_loc(owner: str, repo_name: str, owner_id: str, cursor: Optional[str] = None) -> Tuple[int, int, int, int]:
    """Return (added, deleted, my_commits, total_commits_for_branch)
       Uses pagination (first: 100) over commit history.
    """
    query = '''
    query ($repo_name: String!, $owner: String!, $cursor: String) {
      repository(name: $repo_name, owner: $owner) {
        defaultBranchRef {
          target {
            ... on Commit {
              history(first: 100, after: $cursor) {
                totalCount
                edges {
                  node {
                    ... on Commit {
                      committedDate
                      additions
                      deletions
                      author { user { id } }
                    }
                  }
                }
                pageInfo { endCursor hasNextPage }
              }
            }
          }
        }
      }
    }
    '''
    # We do manual POSTs here to preserve partial cache behavior like the original script.
    query_count('recursive_loc')
    payload = {'query': query, 'variables': {'repo_name': repo_name, 'owner': owner, 'cursor': cursor}}
    resp = _post_graphql(payload)
    j = resp.json()['data']['repository']
    db = j.get('defaultBranchRef')
    if not db:
        # empty/default branch missing
        return 0, 0, 0, 0
    hist = db['target']['history']
    added = 0
    deleted = 0
    my_commits = 0
    for edge in hist['edges']:
        node = edge['node']
        # node might be {...} with additions/deletions and author
        additions = node.get('additions') or 0
        deletions = node.get('deletions') or 0
        author = None
        if 'author' in node and node['author'] and node['author'].get('user'):
            author = node['author']['user'].get('id')
        if author == owner_id:
            added += additions
            deleted += deletions
            my_commits += 1
    total_count = hist.get('totalCount', 0)
    # If there's a next page, recurse
    page = hist.get('pageInfo', {})
    if page.get('hasNextPage'):
        sub_added, sub_deleted, sub_my_commits, _ = recursive_loc(owner, repo_name, owner_id, page.get('endCursor'))
        added += sub_added
        deleted += sub_deleted
        my_commits += sub_my_commits
    return added, deleted, my_commits, total_count

#  Cache management 

def cache_filename(user_name: str) -> str:
    return os.path.join('cache', hashlib.sha256(user_name.encode('utf-8')).hexdigest() + '.txt')


def flush_cache(edges: List[dict], filename: str, comment_size: int) -> None:
    # preserve comment header (if any) and write fresh lines for each repo
    preserved = []
    if os.path.exists(filename) and comment_size > 0:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            preserved = lines[:comment_size]
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(preserved)
        for node in edges:
            repo_name = node['node']['nameWithOwner']
            repo_hash = hashlib.sha256(repo_name.encode('utf-8')).hexdigest()
            f.write(f"{repo_hash} 0 0 0 0\n")


def cache_builder(edges: List[dict], comment_size: int = CACHE_COMMENT_SIZE, force_cache: bool = False) -> List[Any]:
    """Return [loc_add, loc_del, total_loc, cached_flag]
       Edges is the list of repository edges from GraphQL.
    """
    filename = cache_filename(USER_NAME)
    if not os.path.isdir('cache'):
        os.makedirs('cache', exist_ok=True)

    # Read or create cache file
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        # create comment block if requested
        lines = []
        if comment_size > 0:
            for _ in range(comment_size):
                lines.append('This line is a comment block. Edit if you wish.\n')
        with open(filename, 'w', encoding='utf-8') as f:
            f.writelines(lines)

    if len(lines) - comment_size != len(edges) or force_cache:
        flush_cache(edges, filename, comment_size)
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

    cache_comment = lines[:comment_size]
    data_lines = lines[comment_size:]

    # Build a dict repo_hash -> line index and values
    cache_map: Dict[str, Dict[str, Any]] = {}
    for i, line in enumerate(data_lines):
        parts = line.strip().split()
        if len(parts) >= 5:
            repo_hash = parts[0]
            commit_count = int(parts[1])
            my_commits = int(parts[2])
            loc_add = int(parts[3])
            loc_del = int(parts[4])
        else:
            continue
        cache_map[repo_hash] = {
            'index': i,
            'commit_count': commit_count,
            'my_commits': my_commits,
            'loc_add': loc_add,
            'loc_del': loc_del,
        }

    # For each repo edge, compare and update
    for i, edge in enumerate(edges):
        repo_name = edge['node']['nameWithOwner']
        repo_hash = hashlib.sha256(repo_name.encode('utf-8')).hexdigest()
        total_count = 0
        # safe path to nested totalCount
        try:
            total_count = edge['node']['defaultBranchRef']['target']['history']['totalCount']
        except Exception:
            total_count = 0
        cached_entry = cache_map.get(repo_hash)
        if cached_entry and cached_entry['commit_count'] == total_count and not force_cache:
            # no change
            continue
        # update needed: run recursive_loc
        owner, repo = repo_name.split('/')
        try:
            added, deleted, my_commits, repo_total_count = recursive_loc(owner, repo, OWNER_ID)
        except Exception as e:
            log.exception("Error while counting LOC for %s: %s", repo_name, e)
            # try to preserve partial progress
            force_close_file(data_lines, cache_comment)
            raise
        # update or append in data_lines (ensure index i exists)
        if i < len(data_lines):
            data_lines[i] = f"{repo_hash} {repo_total_count} {my_commits} {added} {deleted}\n"
        else:
            data_lines.append(f"{repo_hash} {repo_total_count} {my_commits} {added} {deleted}\n")

    # write back cache
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(cache_comment)
        f.writelines(data_lines)

    # Sum totals
    loc_add_total = 0
    loc_del_total = 0
    for line in data_lines:
        parts = line.split()
        if len(parts) >= 5:
            loc_add_total += int(parts[3])
            loc_del_total += int(parts[4])
    total_loc = loc_add_total - loc_del_total
    cached_flag = True
    return [loc_add_total, loc_del_total, total_loc, cached_flag]


def force_close_file(data_lines: List[str], cache_comment: List[str]) -> None:
    filename = cache_filename(USER_NAME)
    with open(filename, 'w', encoding='utf-8') as f:
        f.writelines(cache_comment)
        f.writelines(data_lines)
    log.warning('Partial cache data saved to %s', filename)

#  LOC query (top-level) 

def loc_query(owner_affiliations: List[str], comment_size: int = CACHE_COMMENT_SIZE, force_cache: bool = False, cursor: Optional[str] = None, edges_accum: Optional[List[dict]] = None) -> List[Any]:
    """Paginate repositories (60 at a time to avoid server timeouts) and return cache_builder totals."""
    if edges_accum is None:
        edges_accum = []
    query_count('loc_query')
    query = '''
    query ($ownerAffiliations: [RepositoryAffiliation], $login: String!, $cursor: String) {
      user(login: $login) {
        repositories(first: 60, after: $cursor, ownerAffiliations: $ownerAffiliations) {
          edges {
            node { ... on Repository { nameWithOwner defaultBranchRef { target { ... on Commit { history { totalCount } } } } } }
          }
          pageInfo { endCursor hasNextPage }
        }
      }
    }
    '''
    r = simple_request('loc_query', query, {'ownerAffiliations': owner_affiliations, 'login': USER_NAME, 'cursor': cursor})
    repo_page = r.json()['data']['user']['repositories']
    edges_accum.extend(repo_page['edges'])
    if repo_page['pageInfo']['hasNextPage']:
        return loc_query(owner_affiliations, comment_size, force_cache, repo_page['pageInfo']['endCursor'], edges_accum)
    return cache_builder(edges_accum, comment_size, force_cache)

#  Archive handling 

def add_archive() -> List[int]:
    """Read cache/repository_archive.txt and return [added_loc, deleted_loc, net_loc, added_commits, contributed_repos]
       Format is the same as produced by the original script.
    """
    path = os.path.join('cache', 'repository_archive.txt')
    if not os.path.exists(path):
        return [0, 0, 0, 0, 0]
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # the original file had a comment header and some footer; this implementation will attempt to
    # skip the first 7 lines and last 3 (as in the PDF) if the file is long enough; otherwise parse as available.
    if len(lines) > 10:
        data_lines = lines[7:-3]
    else:
        data_lines = [l for l in lines if l.strip() and not l.startswith('#')][:]
    added_loc = 0
    deleted_loc = 0
    added_commits = 0
    contributed_repos = 0
    for line in data_lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            # repo_hash total_commits my_commits loc_add loc_del
            my_commits = parts[2]
            loc_add = int(parts[3])
            loc_del = int(parts[4])
            added_loc += loc_add
            deleted_loc += loc_del
            if my_commits.isdigit():
                added_commits += int(my_commits)
            contributed_repos += 1
    return [added_loc, deleted_loc, added_loc - deleted_loc, added_commits, contributed_repos]

#  SVG editing 

def find_and_replace(root: etree._ElementTree, element_id: str, new_text: str) -> None:
    """Robustly find an SVG element by id (namespace-aware) and replace text."""
    # Try xpath which works regardless of default namespace
    result = root.xpath(f"//*[@id='{element_id}']")
    if not result:
        # fallback: try searching for any element with local-name() text
        result = root.xpath(f"//*[local-name() = 'text' and @id='{element_id}']")
    if result:
        el = result[0]
        el.text = str(new_text)


def justify_format(root: etree._ElementTree, element_id: str, new_text: Any, length: int = 0) -> None:
    if isinstance(new_text, int):
        new_text = f"{new_text:,}"
    new_text = str(new_text)
    find_and_replace(root, element_id, new_text)
    just_len = max(0, length - len(new_text))
    if just_len <= 2:
        dot_map = {0: '', 1: ' ', 2: '. '}
        dot_string = dot_map[just_len]
    else:
        dot_string = ' ' + ('.' * just_len) + ' '
    find_and_replace(root, f"{element_id}_dots", dot_string)


def svg_overwrite(filename: str, age_data: str, commit_data: int, star_data: int, repo_data: int, contrib_data: int, follower_data: int, loc_data: List[int]) -> None:
    if not os.path.exists(filename):
        log.warning("SVG file %s not found; skipping.", filename)
        return
    tree = etree.parse(filename)
    root = tree.getroot()
    justify_format(root, 'commit_data', commit_data, 22)
    justify_format(root, 'star_data', star_data, 14)
    justify_format(root, 'repo_data', repo_data, 6)
    justify_format(root, 'contrib_data', contrib_data)
    justify_format(root, 'follower_data', follower_data, 10)
    # loc_data expected: [loc_add, loc_del, net]
    if isinstance(loc_data, (list, tuple)) and len(loc_data) >= 3:
        justify_format(root, 'loc_data', loc_data[2], 9)
        justify_format(root, 'loc_add', loc_data[0])
        justify_format(root, 'loc_del', loc_data[1], 7)
    tree.write(filename, encoding='utf-8', xml_declaration=True)
    log.info('Wrote %s', filename)

#  Other counters 

def commit_counter(comment_size: int) -> int:
    total_commits = 0
    filename = cache_filename(USER_NAME)
    if not os.path.exists(filename):
        return 0
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data_lines = lines[comment_size:]
    for line in data_lines:
        parts = line.split()
        if len(parts) >= 3:
            total_commits += int(parts[2])
    return total_commits

# Main orchestration 

def main():
    log.info('Starting GitHub stats collection for %s', USER_NAME)
    user_data, t_user = perf_counter(user_getter, USER_NAME)
    owner_id, created_at = user_data
    global OWNER_ID
    OWNER_ID = owner_id
    formatter = lambda name, diff: log.info('%-24s %s', name + ':', f'{diff:.4f}s' if diff > 1 else f'{diff*1000:.2f}ms')
    formatter('account data', t_user)

    # Age (hard-coded birthday example from original: 2002-07-05)
    age_data, t_age = perf_counter(daily_readme, datetime.datetime(2002, 7, 5))
    formatter('age calculation', t_age)

    # Loc query: owner + collaborator + organization member
    affiliations = ['OWNER', 'COLLABORATOR', 'ORGANIZATION_MEMBER']
    if DEBUG:
        log.info('DEBUG mode: limiting repository scanning')
    loc_result, t_loc = perf_counter(loc_query, affiliations, CACHE_COMMENT_SIZE, False)
    formatter('LOC (cached)', t_loc)

    commit_data, t_commit = perf_counter(commit_counter, CACHE_COMMENT_SIZE)
    formatter('commit counter', t_commit)

    star_data, t_star = perf_counter(graph_repos_stars, 'stars', ['OWNER'])
    formatter('stars', t_star)

    repo_data, t_repo = perf_counter(graph_repos_stars, 'repos', ['OWNER'])
    formatter('repos', t_repo)

    contrib_data, t_contrib = perf_counter(graph_repos_stars, 'repos', affiliations)
    formatter('contrib', t_contrib)

    follower_data, t_follower = perf_counter(follower_getter, USER_NAME)
    formatter('followers', t_follower)

    # Add archived contributions (if present) - matches the behavior observed in the PDF
    archived = add_archive()
    if archived:
        loc_result[0] += archived[0]
        loc_result[1] += archived[1]
        loc_result[2] += archived[2]
        contrib_data += archived[4]
        commit_data += int(archived[3])

    # Format the loc numbers as strings for the svg
    loc_result_fmt = [f"{x:,}" if isinstance(x, int) else x for x in loc_result[:3]] + loc_result[3:]

    # Write SVGs
    svg_overwrite('dark_mode.svg', age_data, commit_data, star_data, repo_data, contrib_data, follower_data, loc_result)
    svg_overwrite('light_mode.svg', age_data, commit_data, star_data, repo_data, contrib_data, follower_data, loc_result)

    # Print summary
    total_time = t_user + t_age + t_loc + t_commit + t_star + t_repo + t_contrib + t_follower
    log.info('Total function time: %.4f s', total_time)
    log.info('Total GraphQL API calls: %d', sum(QUERY_COUNT.values()))
    for k, v in QUERY_COUNT.items():
        log.info('  %-20s %6d', k + ':', v)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log.exception('Unhandled exception: %s', e)
        sys.exit(1)
