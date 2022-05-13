package com.boot.util;

import javax.servlet.http.HttpServletRequest;

import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.data.domain.Sort.Direction;

public class PageRequestHelper {
	public static PageRequest buildPageRequest(Integer pageNumber, Integer pagzSize, Sort sort) {
		if (pageNumber == null || pageNumber < 1) {
			pageNumber = 1;
		}
		if (pagzSize == null || pagzSize < 1) {
			pagzSize = 20;
		}
		if (sort==null) {
			return PageRequest.of(pageNumber - 1, pagzSize);
		}
		return PageRequest.of(pageNumber - 1, pagzSize, sort);
	}
	public static PageRequest buildPageRequest(HttpServletRequest request,Sort sort) {
		Integer page = 0;
		Integer limit = 10;
		if(StringUtil.isNotBlank(request.getParameter("page")) 
				&& StringUtil.isNotBlank(request.getParameter("limit"))
				&& Integer.parseInt(request.getParameter("page"))>0) {
			page = Integer.parseInt(request.getParameter("page"))-1;
			limit = Integer.parseInt(request.getParameter("limit"));
		}
		if (sort==null) {
			return PageRequest.of(page, limit);
		}
		return PageRequest.of(page,limit,sort);
	}
	public static PageRequest buildPageRequest(HttpServletRequest request,Direction direction, String... properties) {
		Integer pageNumber = 0;
		Integer pagzSize = 10;
		if(StringUtil.isNotBlank(request.getParameter("pageNumber")) 
				&& StringUtil.isNotBlank(request.getParameter("pagzSize"))
				&& Integer.parseInt(request.getParameter("pageNumber"))>0) {
			pageNumber = Integer.parseInt(request.getParameter("pageNumber"))-1;
			pagzSize = Integer.parseInt(request.getParameter("pagzSize"));
		}
		return PageRequest.of(pageNumber, pagzSize, direction, properties);
	}
}
