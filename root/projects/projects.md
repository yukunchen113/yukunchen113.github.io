---
layout: default
title: Experience
permalink: /projects/
---
<ul>
	{% for post in site.posts %}
	{% if post.categories contains "project" %}
    <li>
      <H1><a href="{{ post.url }}">
        {{ post.title }}
      </a></H1>
      - <time datetime="{{ post.date | date: "%Y-%m-%d" }}">{{ post.date | date_to_long_string }}</time>
      {{ post.excerpt }}
    </li>
    {% endif %}
	{% endfor %}
</ul>
