#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html


import gitlab # mamba install python-gitlab or pip install python-gitlab
import argparse
import sys


def get_arguments():
    """Retrieves the arguments of the program.
      Returns: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, usage=
                                     "{0} -h".format(sys.argv[0]))
    parser.add_argument('-i', dest='project_id', type=int, required=True, help='Project ID.')
    parser.add_argument('-k', dest='project_token', type=str, required=True, help='Project token (Create a personal token with a API-writing access).')
    parser.add_argument('-a', dest='author_username', type=str, default='ghost', help='Username of the spamming author')
    parser.add_argument('-r', dest='gitlab_host', type=str, default='https://gitlab.pasteur.fr', help='Gitlab host url')
    args = parser.parse_args()
    return args

def main():
    args = get_arguments()
    # from the work of Jean Curie
    gl = gitlab.Gitlab(url=args.gitlab_host, private_token=args.project_token)
    project = gl.projects.get(id=args.project_id)

    issues_to_delete=[]
    page=1
    # why not doing directly issue.delete() ? Because of a bug in pagination:
    # when doing issue.delete() on the i issue of page j we always try to delete fetched issue i of page 1.
    # To prevent it, we fetch all issue at once then delete them one by one.
    while True:
        print(f"fetching issues, page {page}")
        more_issues = [issue.iid for issue in project.issues.list(
            per_page=50,
            author_username=args.author_username, # only issues od the spammer
            page=page,
        )]
        if len(more_issues) == 0:
            print(f"page empty leaving")
            break
        issues_to_delete += more_issues
        page += 1

    print("Issues to delete: ", [i for i in issues_to_delete])
    for issue in issues_to_delete:
        print(f"deleting issue {issue}")
        project.issues.get(issue).delete()



if __name__ == '__main__':
    main()
