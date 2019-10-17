---
layout: info_sidebar
title: Experience
permalink: /projects/
---
<ul>
	{% for post in site.posts %}
	{% if post.categories contains "project" %}
    <li>
      <a href="{{ post.url }}">
        {{ post.title }}
      </a>
      - <time datetime="{{ post.date | date: "%Y-%m-%d" }}">{{ post.date | date_to_long_string }}</time>
      {{ post.excerpt }}
    </li>
    {% endif %}
	{% endfor %}
</ul>

<h4>Check out more of my projects on my <a id="link" href="https://github.com/yukunchen113">Github</a>!</h4>