package com.boot.controller;

import java.util.List;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;

import com.boot.entity.Dictionary;
import com.boot.repository.DictionaryRepository;
import com.boot.util.PageRequestHelper;
import com.boot.vo.BaseMessage;
import com.boot.vo.MessageHandler;
/**
 * 字典管理
 * 标签
 */
@Controller
@RequestMapping("/private/dictionary")
public class DictionaryController {
	@Autowired
	private DictionaryRepository dictionaryRepository;
	@ResponseBody
	@RequestMapping("/list")
	public BaseMessage<?>  dictionaryList(String name, HttpServletRequest request, HttpServletResponse response) {
		try {
			Pageable pageable = PageRequestHelper.buildPageRequest(request, null);
			Page<Dictionary> pageList = dictionaryRepository.findAll(pageable);
			return MessageHandler.createSuccessVo(pageList.getContent(),"operate successfully",
					(int) pageList.getTotalElements());
		} catch (Exception e) {
			e.printStackTrace();
		}
		return MessageHandler.createFailedVo("操作失败");
	}
	@ResponseBody
	@RequestMapping("/list/list")
	public  BaseMessage<?>  dictionaryListList(HttpServletRequest request, HttpServletResponse response) {
		try {
			List<Dictionary> list = dictionaryRepository.findAll();
			return MessageHandler.createSuccessVo(list,"operate successfully");
		} catch (Exception e) {
			e.printStackTrace();
		}
		return MessageHandler.createFailedVo("操作失败");
	}
	@ResponseBody
	@RequestMapping("/dell")
	public  BaseMessage<?>  dell(Integer id, HttpServletRequest request, HttpServletResponse response) {
		try {
			//判断是否绑定数据
			dictionaryRepository.deleteById(id);
			return MessageHandler.createSuccessVo("operate successfully");
		} catch (Exception e) {
			e.printStackTrace();
		}
		return MessageHandler.createFailedVo("操作失败");
	}
	@ResponseBody
	@RequestMapping("/save")
	public  BaseMessage<?>  save(Dictionary q, HttpServletRequest request, HttpServletResponse response) {
		try {
			dictionaryRepository.save(q);
			return MessageHandler.createSuccessVo("operate successfully");
		} catch (Exception e) {
			e.printStackTrace();
		}
		return MessageHandler.createFailedVo("操作失败");
	}
}
